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
   request errors; the other 63 finish), and `curl -s localhost:8911/health` stays `ok`. There's
   no switch to force a device fault; on the next real one, look for the log line
   `tang-llm: the engine is unhealthy:` and `/health` answering 503.

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

3. **A fresh slot writes its own table entries.** `alloc_slot`/`free_slot` used to rebuild every
   layer's `[u64; MAX_SLOTS]` stab/htab/ktab/vtab table (and MTP's) on the host, upload all of
   them and sync. Now `write_slot_entry(i)` writes slot `i`'s two f32 words per table at `2 * i`
   (`CudaComputeDevice::upload_into_f32_at`, ~2 small copies a layer), the same values the full
   rewrite would (0 for a freed slot); the full `write_slot_tables` + sync runs once at load.
   No sync on GPU 0: the uploads are on the engine's stream, which every reader runs on or forks
   from by event (`ev_fork` for the side stream), and graphs read the tables by address at
   launch. What relied on the sync: only the layer-split peer, whose kernels run on its `Ctx`
   stream while its uploads go through its `CudaComputeDevice` stream. The old code synced only
   GPU 0, so that was a latent race; the peer stage is now synced after its entries are written.
   `TANG_SLOT_TABLE_CHECK=1` reads every table back after each per-entry write, does a full
   rewrite, and asserts the words are identical. GPU checks: `TANG_SLOT_TABLE_CHECK=1
   TANG_BATCH_N=64 tang-llm flash-batch-check $G` (no assert; then without the env:
   `TANG_BATCH_N=64 tang-llm flash-batch-check $G` gives the same PASS line, tolerance numbers
   unchanged from 2026-11-03), and `flash-batch-check` at `TANG_BATCH_N=8` stays bitwise.

4. **One-row steps never take the batch module.** The fault (`CUDA_ERROR_ILLEGAL_ADDRESS` at
   the sync after a 1-row window) came from the `batch_all` experiment (88609b41 dropped it),
   which sent every step through the batch module. The module's dense paths (K-split
   `cx_gemvkb`, Q8X MMA, router) are all gated on `t > MAX_T`, so a short window there falls
   through to the narrow kernels (`gemvk[t-1]`) on the batch module's scratch and int8 staging,
   whose scales sit at `rows * k`, not `MAX_T * k`. That mixed-module path is the likely cause
   (not reproduced here, no GPU). `cx_moe_plan_w` is safe at `n = 11` (always launched with a
   1024-thread block) and `gemv_batch` pads the last launch to 8/16/32 rows inside the scratch.
   The fix is the routing rule, now explicit and guarded: `batch::batch_module(rows, MAX_T,
   batch_rows)`: only 9..=batch_rows rows use the batch module, 1..=8 always use the narrow one
   (`decode_batch`), and `run_staged` refuses a ≤ 8-row window in the batch module with an error
   (which #1 retires as that window's failure) instead of faulting the device. Under
   `serve --batch 64`, concurrency dropping to 1–8 decodes narrow. Test:
   `batch::tests::short_steps_never_take_the_batch_module`. GPU checks: `TANG_BATCH_N=64
   tang-llm flash-batch-check $G` (same PASS and tolerance numbers as 2026-11-03); `TANG_BATCH_N=8`
   bitwise; and `tang-llm serve $G --batch 64`, 64 concurrent requests with staggered
   `max_tokens` (16, 32, …, 1024) so the batch drains through 64 → 9 → 8 → 1 rows, with no fault
   and each finishing request's text equal to the same request run alone (greedy).

## Scheduler for B = 64 (2026-11-04)

### What the load test spent its GPU time on

From `artifacts/gates-1104/serve-c9f064b5.log` (the per-request `batch req:` lines: prefill ms,
TTFT, decode tokens / seconds / windows) and `lt-*.json`, H200, `serve --batch 64`, prompts
200–6000 tokens (avg ~2,600 served), 256–1024 out:

| | C = 8 (24 req, 98 s) | C = 64 cold (192 req, 649 s) |
|---|---|---|
| prompt tokens prefilled | 29.5 k | 501 k |
| prefill GPU time (sum of chunks) | 24 s, **25 %** of wall | 372 s, **57 %** of wall |
| prefill speed inside a chunk | 1,220 tok/s (0.81 ms/token) | 1,350 tok/s (0.72–0.91 ms/token) |
| output tokens | 15.2 k → 155 tok/s | 123 k → 189 tok/s |
| streams decoding, on average | 4.8 | **2.2** |
| decode-window wall per stream token | 31 ms | 12 ms |
| mean TTFT | 15 s | 109 s (p95 505 s, max 631 s) |

So at C = 64 the GPU was busy (57 % prefill, the rest decode windows of ~2 rows) and decode
rows were nearly empty: admissions came out of prefill at ~0.25 a second, each decoded its
~640 tokens in ~8 s, so ~2 rows were ever decoding. The 64-row windows (44 ms for 64 tokens,
`r-sample64.log`) never ran full.

**The ceiling is prefill speed, not the policy.** Wide prefill runs at ~1,350 prompt tok/s.
A request of ~3,100 prompt tokens and ~640 out costs 2.3 s of prefill and, at best (64-row
windows), 0.44 s of decode, so no schedule beats ~640 / 2.74 ≈ **233 output tok/s** on this mix
(280 if decode were free). The 1,000 tok/s aggregate target needs prefill ≥ ~6–7k tok/s
(the wide windows' grouped-expert / GEMM path), and TTFT p95 ≤ 30 s for a cold burst of 64
such prompts needs the same: prefilling all 64 takes 64 × 2.3 s = 147 s at 100 % of the GPU.

### The policy (`crate::batch::prefill`, used by `flash/serve_batch.rs`)

- **Order: shortest-first with aging and a deadline.** Key = remaining prompt tokens −
  `TANG_SERVE_AGE_TOKENS` (400) × seconds waited; past `TANG_SERVE_PREFILL_DEADLINE_S` (20 s) a
  prompt goes ahead of every younger one, oldest first. A cached agent turn still jumps a cold
  prompt; a 6000-token prompt can't sit 500 s behind short ones.
- **Prefill share depends on decode fill.** While prompts wait and fewer than
  `TANG_SERVE_DECODE_FILL` (0.8) × batch rows decode, prefill gets `TANG_SERVE_PREFILL_FILL_SHARE`
  (0.85) of GPU time; once rows are full, `TANG_SERVE_PREFILL_SHARE` (0.5, the old split). A
  decode window of `dt` pays off `dt × share / (1 − share)` seconds of prefill.
- **Chunk size scales with free GPU time.** Decode credit banks (up to one max chunk); a chunk
  is as many tokens as the credit buys at the measured prefill rate (EMA), between
  `TANG_SERVE_PREFILL_SLICE` (512 with wide windows, 64 without) and
  `TANG_SERVE_PREFILL_MAX_CHUNK` (2048 / 256). With nothing decoding: `TANG_SERVE_PREFILL_BUDGET`
  (2048 / 256). The server logs the policy in effect at its first step.
- The old behaviour is `TANG_SERVE_PREFILL_SLICE=256 TANG_SERVE_PREFILL_MAX_CHUNK=256
  TANG_SERVE_PREFILL_FILL_SHARE=0.5 TANG_SERVE_AGE_TOKENS=0 TANG_SERVE_PREFILL_DEADLINE_S=0`.

### Expected effect (simulated)

`batch::prefill::tests` runs the load closed-loop on a simulated engine with the logged costs
(prefill 1/1350 s a token + 4 ms a chunk, a decode window 10 ms + 0.53 ms a row). The simulation
reproduces the old run's shape (151 tok/s, p95 680 s at C = 64). Old → new:

| | agg tok/s | TTFT p50 | TTFT p95 | per-stream tok/s | rows/window |
|---|---|---|---|---|---|
| C = 64, old | 151 | 39 s | 680 s | 40 | 3.6 |
| C = 64, new | **246** | 46 s | **122 s** | 5 | 36 |
| C = 8, old | 155 | 4.8 s | 41 s | 41 | 3.5 |
| C = 8, new | **191** | **2.0 s** | **12 s** | 28 | 5.9 |

The trade is per-stream speed: with prefill at 85 % of the GPU while rows are short, a running
stream gets fewer windows a second (C = 64: 5 tok/s a stream while 60+ prompts queue; it rises
as rows fill and the share drops to 0.5). `TANG_SERVE_PREFILL_FILL_SHARE` is the dial
(`cargo test -p tang-llm --lib batch::prefill::tests::sweep -- --ignored --nocapture` prints the
sweep: 0.6 gives C = 64 182 tok/s / p95 245 s / 30 tok/s a stream). Neither target (≥ 1,000 tok/s,
p95 ≤ 30 s at C = 64) is reachable at 1,350 prefill tok/s; the next lever is wide prefill speed.

### Verify

On the pod (`/workspace/tang-serve`, model `/workspace/m/`, under `flock /workspace/gpu.lock`;
the client is the untracked `artifacts/lt/load.py`, copied over first with
`scp -P 16399 artifacts/lt/load.py root@152.236.142.239:/workspace/load.py`):

```
cargo build --release -p tang-llm --no-default-features --features cuda
G=$(ls /workspace/m/*.gguf | head -1)
target/release/tang-llm serve $G --batch 64 --ctx 32768 > serve-sched.log 2>&1 &
# wait for "listening on", then, for each C:
# serve listens on 8911 by default (the "listening on" line has the port):
for C in 8 64; do python3 /workspace/load.py $C 8911 3 > lt-sched-$C.json; done
grep "batch prefill policy" serve-sched.log
```

Compare `agg_tps`, `ttft_p50/p95`, `stream_tps_p50` against `artifacts/gates-1104/lt-8.json` and
`lt-64cold.json`; rerun C = 64 with `TANG_SERVE_PREFILL_FILL_SHARE=0.6` to place the dial.

## bf16 GDN state (2026-11-04, H200 ru6a6zybgtu80q)

`TANG_CODER_GDN_STATE=f32|bf16|f16|bf16sr` (default **f32**). A 16-bit mode stores each layer's
recurrent state as 2-byte elements: half the per-slot GDN memory (75 → 38 MB at 36 layers) and
half the state traffic. The recurrence still runs in f32 registers. The rounding (`gst_rt`)
happens **after every token's update**, in registers, not only when the window stores the
state. Rounding only at the store made a token's state depend on its window's width:
`flash-tcheck` t1 vs t8 differed in every logit and `flash-spec-test` FAILed (every mode
DIFFERS). With per-token rounding a T=8 window computes what eight T=1 windows compute, so
tcheck and spec stay bitwise. The kernels are `fl_gdn_conv_step(_seg)` and `fl_gdn_wide`
(prefill). The slot tables hold addresses only, so they don't change. `slot_bytes`, slot
allocation, the VRAM cap and save/restore follow the mode. A state saved under one mode is
rejected under the other. `TANG_CODER_GDN_RBITS=n` is a diagnostic: f32 storage with n
significand bits rounded off per token.

**Gate tool:** `flash-state-kl` (wide prefill, then 2,000 greedy steps). With `--dump`, an f32 run writes
its tokens and logits. A second process under the other mode runs teacher-forced on those
tokens and prints mean/max KL(f32 ‖ mode) and top-1 agreement per 512-token bucket. Prompts:
the AA explain/refactor/write (~1.24k tokens) and an 8,000-token review of `flash/cli.rs`.

| prompt | positions | bf16 mean KL | bf16 top-1 | f16 mean KL | f16 top-1 | floor (RBITS=2) KL | floor top-1 |
|---|---|---|---|---|---|---|---|
| explain | 0–512 | 5.15e-3 | 97.66% | 3.57e-3 | 98.44% | 3.28e-3 | 98.24% |
| | 512–1024 | 4.53e-3 | 98.05% | 3.05e-3 | 98.05% | 3.03e-3 | 98.83% |
| | 1024–1536 | 5.25e-3 | 97.27% | 4.34e-3 | 96.68% | 4.35e-3 | 96.68% |
| | 1536–2000 | 4.63e-3 | 97.84% | 3.40e-3 | 97.41% | 2.81e-3 | 97.84% |
| | **all** | **4.90e-3** | **97.70%** | **3.59e-3** | **97.65%** | **3.38e-3** | **97.90%** |
| refactor | 0–512 | 1.27e-3 | 99.80% | 1.33e-3 | 99.61% | | |
| | 512–1024 | 1.93e-3 | 98.24% | 1.85e-3 | 98.44% | | |
| | 1024–1536 | 1.33e-3 | 99.22% | 1.06e-3 | 99.80% | | |
| | 1536–2000 | 1.89e-3 | 99.57% | 1.43e-3 | 99.35% | | |
| | **all** | **1.60e-3** | **99.20%** | **1.42e-3** | **99.30%** | | |
| write | 0–512 | 3.34e-3 | 98.44% | 2.66e-3 | 98.05% | | |
| | 512–1024 | 1.55e-3 | 99.41% | 1.62e-3 | 99.41% | | |
| | 1024–1536 | 2.47e-3 | 99.22% | 2.22e-3 | 99.22% | | |
| | 1536–2000 | 5.83e-4 | 99.78% | 7.91e-4 | 99.78% | | |
| | **all** | **2.02e-3** | **99.20%** | **1.85e-3** | **99.10%** | | |
| 8k review | 0–512 | 4.10e-3 | 97.66% | 2.28e-3 | 98.05% | 2.30e-3 | 98.24% |
| | 512–1024 | 3.66e-3 | 98.44% | 2.20e-3 | 98.83% | 2.17e-3 | 99.41% |
| | 1024–1536 | 3.47e-3 | 97.66% | 1.85e-3 | 98.83% | 1.75e-3 | 98.63% |
| | 1536–2000 | 4.83e-3 | 97.63% | 2.93e-3 | 98.06% | 3.08e-3 | 98.92% |
| | **all** | **3.99e-3** | **97.85%** | **2.30e-3** | **98.45%** | **2.31e-3** | **98.80%** |

**The task's bars (mean KL ≤ 1e-3, agreement ≥ 99.5%) can't be met by any state that isn't
bitwise f32. This engine has a perturbation floor.** f32 against its own reference is exactly 0 (deterministic).
Rounding off just the **2 lowest** of 23 significand bits (≈5e-7 relative) already gives
2.3–3.4e-3 and 97.9–98.8% agreement, the same as f16 or bf16. Earlier sweeps on explain, with rounding at the
window store only, show the same: RBITS 2 / 6 / 13 gave 2.75 / 3.20 / 3.04e-3, f16 2.88e-3, bf16 3.50e-3. Any
change to the state, however small, flips near-tied decisions downstream (top-10 of 512
experts, greedy near-ties), and the divergence saturates at that level. It doesn't scale with
the size of the error. The f32 batch path already accepted at B=64 shows the same kind of
divergence: `flash-batch-check` N=64 tolerance is 99.10% token agreement on f32.

**Measured against that floor:**
- **f16 matches the floor:** 8k 2.30e-3 vs 2.31e-3; explain 3.59 vs 3.38e-3.
- **bf16 is 1.45–1.7× above it:** explain 4.90e-3, 8k 3.99e-3, with 0.3–1 point lower agreement.
- **Drift:** neither mode's KL trends upward across the 512-token buckets, on any prompt (8k: 4.1 / 3.7 /
  3.5 / 4.8e-3 for bf16; the last bucket's rise is within the floor's own spread, 2.3 → 3.1e-3).
- **Stochastic rounding** (`bf16sr`, rounding at the window store) didn't help: explain 3.58e-3, 8k 2.23e-3, vs
  RNE's 3.50 / 2.18e-3.

**Gates under bf16 / f16:**

| gate | f32 | bf16 | f16 |
|---|---|---|---|
| flash-tcheck t1 vs t8 / t8-verify | 0 / 0 differ | 0 / 0 | 0 / 0 |
| flash-spec-test (all modes SAME, T 0 and 0.8) | PASS | PASS | PASS |
| flash-batch-check N=8 | PASS bitwise | PASS bitwise | PASS bitwise |
| flash-batch-check N=64 (tolerance) | 99.10% agree, mean \|dlogit\| 0.124 | 99.14%, 0.132 | — |
| needle (passphrase at 10% of 8k tokens) | found | found | found |

(batch-check compares a mode's batch output with the same mode alone: the batch kernels stay
row-independent under 16-bit state. Its tolerance numbers at N=64 are the same as f32's.)

**Speed** (explain prompt, same binary, H200). Quick = `TANG_BATCH_QUICK` graph replay at a
64-token prompt. GDN = `gdn_conv_step` ms per step from an eager `TANG_CODER_PROF` run.

| | f32 | bf16 | Δ |
|---|---|---|---|
| B=8 quick ms/step | 9.06 | 8.94 | −0.12 |
| B=32 quick ms/step | 19.15 | 18.78 | −0.37 |
| B=64 quick ms/step | 30.24 | 29.63 (f16 29.43) | −0.6 |
| B=8 GDN ms/step | 0.72 | 0.67 | −0.05 |
| B=32 GDN ms/step | 2.19 | 1.89 | −0.30 |
| B=64 GDN ms/step | 4.29 | 3.68 | −0.61 |
| B=64 batch-check ms/step (ctx ~1.2k) | 38.85 | 38.04 | −0.8 |
| T=1 window (eager bench, median) | 7.47 ms | 7.53 ms | +0.06 |
| T=3 MTP window (median) | 5.79 ms | 5.84 ms | +0.05 |

The win is about 30% of the predicted 2.1 / 1 ms. That prediction assumed the f32 step sits at the
state-bandwidth floor. It doesn't: B=64 moves 36 × 64 × 4 MB ≈ 9.4 GB a step, ≈2 ms at
4.8 TB/s, against 4.29 ms measured. The rest is the per-token barriers and reductions, which
16-bit storage doesn't change. At T=1 the conversion is slightly slower (11.0 vs 10.6 µs a
layer). A vectorized state load/store (2 × bf16 per lane) and fewer barriers per token are
the next lever if GDN matters again.

**Decision: default stays f32.** bf16 fails the stated bars and is measurably above the
floor. f16 is at the floor, passes every bitwise gate and gives the same speed (B=64 29.43
ms). It's the mode to flip if the bars are restated relative to the floor, for example
"mean KL ≤ 1.2× an RBITS=2 run, no upward trend". The win is small: 0.6 ms (2%) at B=64, none
single-stream.

## Load-test bug sweep (2026-11-04, coder-serve2)

Found in `serve --batch 128 --ctx 131072` (`artifacts/gates-1104/serve-b128.log` in coder-serve).

- **(a) `--batch` clamped twice (128 → 27 → 26).** `flash serve` counts slots before and after
  building the batch and wide windows; the second pass clamped the already clamped value and
  warned again. The first count is now provisional and silent (it sizes the batch windows); the
  count that sticks is `--batch` against the count after the windows, capped by the provisional
  one, with one warning: `--batch 128 clamped to 26: 27 sequence slots fit … after the batch and
  wide windows`. The drop to 26 itself is real: the windows took 2.8 GB (96.6 → 93.8 GB free),
  most of a 3.30 GB slot at 131k. (`fit_slots` adds back what live slots hold, and none exist
  at startup, so it wasn't counting its own allocation.) Test:
  `batch::tests::startup_clamps_once_from_the_request`.
- **(b) The phantom `req 128: prompt 0 … state saves 7.0–7.5 s` after every completion.** The
  worker calls `Backend::after` for batched requests too, and `FlashServe::after` printed the
  one-stream path's stale `line`, whose `snap_ms` every batched prefix-state save kept adding
  to: 7 s was the running total over all requests, not work per completion. The slot-0 running
  snapshot after it never ran (slot 0 holds no tokens under `--batch`; the log has no
  `saved the running state` line). Not a throughput bug. `after` now returns for batched
  requests (it logs only an error), and the `batch req:` line carries the request's own
  `state saves N ms` (part of its prefill time). No unit test (FlashServe needs a model).
- **(c) Over-context requests.** One rule, `batch::token_limit`: a prompt with no room for a
  generated token (`prompt + 1 >= --ctx`) is refused at admission, before a slot is taken, with
  `context_length_exceeded: the prompt is N tokens; the context window is C`; otherwise
  `max_tokens` is cut so the sequence ends inside the window and the request finishes with
  `finish_reason: "length"` there. Tests (`server::context_limits`, a fake batching engine): the
  refused request never joins the batch, a stream reaching the window stops with `length` while
  the stream beside it finishes, and the admission error answers HTTP 400 to a non-streaming
  client. A streaming client has its 200 headers already, so it gets the same message as an SSE
  `error` event (the tokenizer runs on the worker for the flash engine, so the check can't run
  before the response starts). Real probe (`--ctx 4096` keeps prompts small):

  ```
  tang-llm serve $G --batch 8 --ctx 4096 --port 8921 &
  python3 - <<'PY'
  import json, urllib.request as u, concurrent.futures as cf
  def chat(text, mt, stream=False):
      b = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": mt,
                      "temperature": 0, "stream": stream}).encode()
      r = u.Request("http://127.0.0.1:8921/v1/chat/completions", b, {"content-type": "application/json"})
      try:
          with u.urlopen(r) as f: return f.status, json.load(f)
      except u.HTTPError as e: return e.code, json.load(e)
  print(chat("word " * 6000, 16))             # 400, context_length_exceeded
  with cf.ThreadPoolExecutor(4) as ex:         # near the window + max_tokens past it, beside 3 normal
      fs = [ex.submit(chat, "word " * 3900, 1000)] + [ex.submit(chat, "count to 50", 200) for _ in range(3)]
      for f in fs: s, j = f.result(); print(s, j["choices"][0]["finish_reason"], j["usage"])
  PY
  curl -s localhost:8921/health
  ```
  Expect the first 400, the near-window request `200 length` with `prompt + completion < 4096`,
  the other three finishing normally, `/health` ok. **Result** (H200, `artifacts/spec/probe2.txt`;
  the near-window prompt also asks to count to 3000 so it can't stop on its own): 400
  `context_length_exceeded: the prompt is 6009 tokens; the context window is 4096`; near-window
  `200 length`, 3929 + 166 = 4095 tokens; the three others `stop` at 170; `/health` ok.
- **(d) `TANG_SERVE_DETERMINISTIC=1`.** Batched greedy ≠ solo greedy past 8 rows (7/64
  identical in the drain test) is the batch module's tolerance path (K-split GEMVs, grouped
  experts, MMA) and stays. The switch doesn't build the batch module and runs every decode
  window at ≤ `MAX_T` (8) rows on the narrow, bitwise module (`batch::window_width`), more
  windows a step, so a stream's greedy output is independent of what it's batched with.
  Throughput cost (decode-heavy load below, `--batch 64`, no drafts): C = 8 617 vs 614 tok/s
  (no change: ≤ 8 streams were narrow anyway), C = 32 630 vs 851 (−26 %), C = 64 620 vs 987
  (−37 %); per-stream p50 at C = 64 10.7 vs 18.1 tok/s.
- **GPU checks of (a)/(b)**: `serve --batch 128 --ctx 131072` now logs one line, `--batch 128
  clamped to 26: 27 sequence slots fit … after the batch and wide windows`
  (`artifacts/spec/serve-clamp.log`); no `req N: prompt 0` lines in any batched serve log (the
  only `req` line left is the refused request's error).
- **Also found: `--mtp` was ignored on the qwen3next serve path** (`CoderOpts` built with
  `..Default::default()`), so `serve --mtp` ran suffix drafts with no MTP head. Fixed.

## Spec decode under batching (2026-11-04, coder-serve2)

Before: `serve --batch N --mtp` capped every window at `MAX_T` = 8 rows (no batch module was
built with MTP) and shared the spare rows round-robin, so past 8 streams nothing drafted and
past 8 every stream waited for ⌈N/8⌉ narrow windows.

**Engine** (Rust control flow; no kernel changes):

- `verify_batch` runs past 8 rows in the batch module: its staging, scratch and kernels, keep
  counts in its row record, and the per-segment commit replayed there. Eager by default (row and
  segment counts change every step; `TANG_SERVE_SPEC_GRAPHS=1` captures graphs, which measured
  no faster). The batch module's one shared `proj` holds only the last layer's projections
  when the commit replays every GDN layer, so verify windows write per-layer projection buffers
  instead (`batch_gproj`, rows × `w_in.rows` per GDN layer, allocated on first use). Without
  that the first runs accepted 23 % of drafts and diverged at token ~3.
- `mtp_groups_wide`: MTP passes over any number of rows, read from the window's own hidden
  rows (batch or wide scratch `x`), 8 rows a pass (the MTP kernels' width), long groups in
  pieces with only the last chained. Batch-module decode and verify windows fill every slot's
  MTP K/V and draft its chain; wide prefill windows fill the MTP K/V of their positions too
  (before, a wide-prefilled prompt's MTP K/V was stale; `TANG_SERVE_WIDE_MTP=0` turns it off).
- `Coder::slot_mtp`: skip the MTP pass after slot windows (and clear those chains).

**Scheduler** (`crate::batch::spec`, pure and tested on a simulated engine):

- D-Cut (arXiv 2607.14647): a stream's chain position `k` scores `∏_{j≤k} p_j` (head
  probabilities); every position in the window is ranked, the best `budget` verified in the rows
  the streams leave (`width − streams`; width = the batch module's rows), each stream's pick a
  prefix. Sampled streams draft nothing.
- Budget, TableTune-style (arXiv 2610.38968): step ms per (streams, draft rows, MTP on), EMA
  cells measured as the server runs, a decayed least-squares fit `a + b·rows + c·mtp_rows` for
  unseen cells, a prior before any; `α` = accepted / predicted (EMA). It picks the budget that
  maximises `(streams + α·Σ top scores) / ms` among `⌈ρ·streams·k⌉`, ρ ∈ {¼, ½, ¾, 1}, against
  **plain** decoding (no MTP pass at all = today's path). While plain wins it runs the head one
  step in 64 (and verifies one draft a stream then) to keep `α` and the scores fresh.
- `TANG_SERVE_SPEC=off|auto|<rows>` (default **auto** since 2026-11-05, see the serve numbers below), `TANG_SERVE_BATCH_ROWS` (batch
  module rows, default `--batch` rounded up to 32), `TANG_SERVE_SPEC_LOG=1` (a line per window;
  a `tang-llm: spec:` summary prints every 30 s regardless).

**Gate** (`tang-llm flash-spec-batch-check $G --mtp mtp.gguf --ids-file explain.ids -n 128`,
H200, trained head `seg-3200b/step-003200`, `TANG_CODER_MTP_K=2`; N sequences of the same prompt
cut to different lengths, each `-n` tokens; budget 0 = `decode_batch`, one row a sequence,
MTP on; logs in `artifacts/spec/`):

| N (module rows) | budget 0 | best fixed budget | auto | identical to budget 0 |
|---|---|---|---|---|
| 2 (narrow, ≤ 8 rows) | 345 tok/s | 6: 484 (1.40×), 63 % accepted | – | 2/2 (bitwise class) |
| 8 (32) | 697 | 32: 736 (1.06×), 68 % | 732 (1.05×) | 2/8, 54 % of tokens (budget 0 narrow, spec in the batch module: different class) |
| 16 (32) | 796 | 32: 952 (1.20×), 88 % | 936 (1.18×) | **16/16** |
| 32 (64) | 1,031 | 32: 1,181 (1.15×), 88 % | 1,256 (1.22×, plain) | **32/32** |
| 64 (128) | 1,273 | 64: 1,176 (0.92×), 91 % | 1,623 (1.28×, plain) | 16/64, 56 % (> 64-row windows leave the ≤ 64-row Q8X MMA class) |

Within a width class spec decode is lossless: every budget gives budget 0's tokens exactly
(N = 2 narrow, N = 16 and 32 in the batch module). Across classes (8-row narrow vs batch module,
≤ 64 vs > 64 rows) tokens drift at the tolerance path's usual rate.

**Serve** (`serve --mtp mtp.gguf --batch 64 --ctx 32768`, decode-heavy closed loop: prompts
200–800 tokens, 512–1024 out, 10 % sampled at T 0.8, R = 2 a worker; `load_dec.py`,
`artifacts/spec/lt-*.json`). Aggregate tok/s (per-stream p50):

| C | off (plain batch windows) | auto | fixed 32 rows | deterministic (off) |
|---|---|---|---|---|
| 8 | **614** (89) | 593 (84) | 540 (88) | 617 (89) |
| 16 | **677** (48) | 652 (46) | 655 (56) | – |
| 32 | **851** (30) | 800 (29) | 785 (30) | 630 (22) |
| 64 | **987** (18) | 892 (16) | – | 620 (11) |

Drafts were accepted at 83–95 % in serve too, but no setting beat plain windows: at serve load
the drafting steps cost more than in the engine check (prefill chunks interleave, 10 % of
streams are sampled and draft nothing, the extra MTP passes after every window and over every
wide-prefilled prompt), and auto, once it measures plain steps, settles on plain and pays only
its probes and the MTP head's prefill fill (3–10 %). The first auto run (before the tuner
measured plain steps) drafted throughout and was 10 % under plain. Hence `TANG_SERVE_SPEC`
defaults to **off**; `auto` is there for loads with fewer, longer greedy streams.

**Where the gain goes.** Acceptance is high (88–99 % at k = 2) but the MTP passes are 8 rows
each, with a sync: at 32 streams a step runs ~8 of them, ~20 % of the step (budget 0 with MTP
on 1,031 tok/s vs plain 1,256). Drafting pays at ≤ 16–32 streams (1.15–1.4×) and loses at 64;
auto lands on the better of the two at each N (it picks plain at 32 and 64 here, because a
plain step costs less than a spec step's extra tokens buy). The next lever is the MTP module
built for the batch module's width (one pass per step instead of ⌈rows/8⌉ × k), which is kernel
work; `TANG_CODER_MTP_K=3` didn't help at the room these N leave.

## Known issues (2026-11-04)

- **`flash-wide-check` at 128 / 1000 tokens misses the 1e-3 KL bar** with FA2 attention (KL
  1.15e-2 at 1000 on 4133d39a). Investigated 2026-11-04 (frog/coder-fa2fix): **no FA2 indexing,
  masking, split or merge bug; the bar can't be met at these lengths by any non-bitwise stage.**
  - `TANG_FA2_CMP=1` runs split-K on the same window after FA2 and prints, per layer, the relative
    L2 difference of the attention outputs: bf16 FA2 0.9-3.2e-3, flat across rows (no tile-edge or
    first-tile pattern), the same at 128 / 1000 / 3000 tokens and 1 / 2 / 4 key splits.
  - Perturbation control: split-K (bitwise with narrow when every other wide switch is off) with
    each attention output scaled by 1 ± 1e-7 (`TANG_ATTN_JITTER`, about one f32 ulp) gives KL
    **2.7e-2 at 128 and 6.2e-3 at 1000**; 1e-6 / 1e-5: 2.6e-2 / 2.8e-2 at 128, 1.2e-2 / 1.3e-3
    at 1000. The refactor prompt cycled to 128 / 1000 tokens has teacher-forced steps where
    the logits jump on any last-bit change (probably MoE top-10 near-ties; not pinned down).
    At 3000 / 8000 (third+ copy of the prompt) the same perturbations stay under 1e-5.
  - So FA2 now runs on fp16 tensor cores instead of bf16 (Q, P and the bf16 K/V tiles, which
    fp16 holds exactly): attention error vs split-K 8x lower (1.1-4.2e-4 rel L2), wide prefill
    -0.4% at 3000 and -1% at 8000. KL unchanged, as the control predicts.

  KL(narrow || wide), refactor ids, 2048-row windows, teacher-forced max over 8 steps (H200):

  | config | 128 | 512 | 1000 | 3000 | 8000 |
  |---|---|---|---|---|---|
  | split-K, other wide switches off¹ | 0 | 0 | 0 | 0 | — |
  | split-K + jitter 1e-7, switches off | 2.7e-2 | — | 6.2e-3 | — | — |
  | FA2 bf16, switches off | 2.7e-2 | 5.7e-4 | 1.5e-2 | 7.7e-6 | 2.1e-6 |
  | FA2 fp16, switches off | 2.2e-2 | 7.4e-4 | 1.3e-2 | 1.2e-5 | 4.1e-6 |
  | defaults, FA2 bf16 (before) | 3.4e-2 | 1.7e-4 | 7.9e-3 | 9.0e-7 | 3.1e-6 |
  | defaults, split-K | 6.5e-2 | 1.3e-4 | 9.1e-3 | 9.7e-7 | 3.4e-6 |
  | defaults, FA2 fp16 (after) | 3.1e-2² | 6.2e-4 | 9.5e-3 | 1.5e-6 | 1.5e-6 |

  ¹ `TANG_CODER_WGEMM=0 TANG_CODER_GDNW=0 TANG_CODER_SGEMM=0`: since afb81919 the warp-per-column
  GDN and the router GEMM are also non-bitwise, so `WGEMM=0` alone is no longer bitwise
  (split-K + `WGEMM=0`: 1.9e-2 at 128, 3.1e-2 at 1000).

  ² next greedy token differs at 128 (bf16: equal), the same last-bit sensitivity. Rows from H200
  (tensor-core pod) and H100 (fa2fix pod) agree exactly; logs in artifacts/fa2fix/. tcheck t1/t8
  bitwise, spec-test PASS (H100).

  What would make the gate meaningful at short context: compare against the jitter control
  (fail only when KL is well above what a 1e-7 perturbation gives at the same length), or use
  prompts that aren't a cycled copy. Not done here.
- **`flash-tcheck` panics at `cli.rs:1913`** (index out of bounds on an empty per-layer
  capture) after printing `t1 vs t8: logits differ in 0`, on 4133d39a too. The bitwise logits
  line is the gate; the per-layer table needs a capture the Coder engine doesn't fill.

## Batch-width MTP passes (2026-11-05, coder-spec3)

The ~20 % spec overhead under batching was the MTP head run `MAX_T` = 8 rows a pass, with a host
sync each: ⌈rows/8⌉ first passes plus ⌈streams/8⌉ per chain step, each a full MoE layer and the
151k-row head (16 passes a step at 64 streams, k = 2). `enable_batch` now also builds the MTP
layer at the batch module's width (`MtpWide`: `cx_mtp_in`/`cx_mtp_pick` compiled with
`MTP_ALT = rows`, its own `mh`/`emb`/`cat`); `mtp_groups_wide` packs up to that many rows a pass
and runs it through the batch module's kernels (`mtp_pass_batch`, on the batch scratch), each
chain step over every chained stream at once (narrow under 9 rows). A step's MTP work is now
one pass plus one per chain step. `TANG_SERVE_MTP_WIDE=0` keeps the old 8-row passes. Wide
prefill's MTP K/V fill uses the same passes (64 rows instead of 8), which speeds spec-off serve
too. The tuner also skips the MTP pass outright when a window has no spare rows.

Gate (`flash-spec-batch-check`, 16 streams, k = 2, rows 32, H200): budget 0 849 (was 815), 16:
1099 tok/s 1.29× (was 998, 1.22×). Budget 0's tokens are identical with either MTP setting
(hash d6e82b51531bbf25). On explain one sequence of 16 diverges at budgets ≥ 16 at token ~110
with wide MTP. That's the batch module's tolerance class (different drafts give different
window compositions), not a corruption: the old narrow MTP diverges the same way on `write`
(15/16 at budgets 8 and 16), and refactor is 16/16 at every budget both ways.

Serve (`load_dec.py`, R = 4 after a warm-up run, `--batch 64 --mtp`, H200), aggregate tok/s
(per-stream p50):

| C | before: off | before: auto | after: off | after: auto |
|---|---|---|---|---|
| 8 | 656 (86) | 653 (85) | 667 (87) | 665 (86) |
| 16 | 702 (47) | 725 (57) | 714 (47) | **785** (63) |
| 32 | 896 (30) | 900 (31) | 923 (31) | **985** (39) |
| 64 | 1103 (18) | 1101 (18) | 1145 (19) | 1146 (19) |

Prompt-heavy mix (`load.py` 64, R = 3): before off 486 / auto 491, after off 543 / auto 551,
0 errors throughout. Hence `TANG_SERVE_SPEC` defaults to `auto`.

## int8 K/V cache (2026-11-05, not yet measured on a GPU)

`TANG_CODER_KV=bf16|int8` (default **bf16**; the switch off compiles the same PTX as before,
byte for byte, so every existing gate is unchanged). What one slot holds at `--ctx 131072`:

| part | bytes | share |
|---|---|---|
| attention K/V, 12 layers × 2 × 131072 × 2 kv heads × 256 × bf16 | 3.22 GB | 90% |
| MTP layer K/V (same shape, one layer) | 0.27 GB | 7.5% |
| GDN state, 36 layers × 32 × 128 × 128 f32 (+ 0.1 MB conv history a layer) | 0.08 GB | 2.2% |
| **total** (`slot_bytes`) | **3.57 GB** | |

K/V is what dominates, so it's what int8 compresses (the GDN state is 2%: `TANG_CODER_GDN_STATE=bf16`
would save 38 MB a slot, under one slot in total, and stays f32). Under `int8` each position of a
layer's K (or V) cache is one 528-byte row: 2 heads × 256 int8 codes, the two heads' f32 scales
(absmax / 127, round-half-even codes, `fa2::kv8_quant`), zero padding to 16 bytes. That's 0.52 of
a bf16 row, and rows stay contiguous a position, so positional save/restore, the tree-commit
copies and `slot_bytes` / `fit_slots` / the `--batch` clamp only change their row size
(`ck::kv_row_bytes`). The MTP layer's K/V use it too. Per slot at 131k: 1.88 GB; where 24 bf16
slots fit on the production H200 (~86–89 GB free past the 4 GB headroom), 45–47 int8 slots fit,
so `--batch 48` clamps to 44–46 (one slot stays spare) unless `TANG_SLOT_HEADROOM_GB` is lowered.

Kernels (dequantize on load, no extra pass): `cx_attn_prep` quantizes each head as it writes
(a block max over the head's 256 values); `cx_attn_part` (split-K decode) loads a lane's 8 codes
as one 8-byte word and multiplies by the row's scale; `cx_attn_fa2` lands the int8 tiles with
`cp.async` in a double-buffered staging area (`FA_SMEM` + 32 KB: 134,144 bytes) and its existing
per-tile conversion pass writes code × scale into the fp16 tiles the MMAs read.

CPU tests (`fa2::tests`): the quantization rule, the row/slot/smem arithmetic (24 bf16 slots →
46+ int8 on the same memory), and the FA2 model on int8 K/V against exact f64 attention at head
dim 256 with outlier K channels (int8 max 4.4e-2 / mean 4.7e-3 against bf16's 6.9e-3 / 7.2e-4 on
that deliberately sharp synthetic softmax; the model gates below are the real bar).

Gate tools added: `flash-batch-check` with `TANG_BATCH_REF_DUMP=F` / `TANG_BATCH_REF=F` compares
one configuration's batched outputs against another process's (batch-check's tolerance rule:
≥ 99% tokens agree given the same history, mean |dlogit| ≤ `TANG_BATCH_TOL`, default 0.2);
`flash-state-kl --ref` now also prints the greedy prefix length (the first teacher-forced
argmax that leaves the reference is where free-running greedy would diverge) and mean / max
|dlogit|, and `--dump` prints the tokens. Pod gates: switch off bitwise against the base build
(batch-check checksums at N = 1 and 8, a state-kl dump byte for byte); int8's own tcheck, spec,
tree, batch8, spec-batch and wide-check 3000 / 8000 (same bars); int8 against bf16 on
batch-check N = 64, state-kl explain / refactor / write / an 8k review (greedy ≥ 64 tokens), and
a ~110k-token needle at depths 10% and 50%; then `serve --mtp --ctx 131072 --batch 48 --kv-ttl
3600` decode-heavy and mix at C = 48, with the slot count and VRAM.
