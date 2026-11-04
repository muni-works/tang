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
  the other three finishing normally, `/health` ok. Result in "Spec decode under batching" below.
- **(d) `TANG_SERVE_DETERMINISTIC=1`.** Batched greedy ≠ solo greedy past 8 rows (7/64
  identical in the drain test) is the batch module's tolerance path (K-split GEMVs, grouped
  experts, MMA) and stays. The switch doesn't build the batch module and runs every decode
  window at ≤ `MAX_T` (8) rows on the narrow, bitwise module (`batch::window_width`), more
  windows a step, so a stream's greedy output is independent of what it's batched with.
  Throughput cost: a step of N streams becomes ⌈N/8⌉ narrow windows instead of one batch
  window; measured below.

## Known issues (2026-11-04)

- **`flash-wide-check` at 1000 tokens misses the 1e-3 KL bar**, on 4133d39a too (KL 1.15e-2;
  greedy tokens still equal). Bisected with the wide-window switches on 4133d39a (H200,
  refactor ids, 2048-row windows, teacher-forced max KL over 8 steps):

  | `TANG_CODER_ATTN` | `TANG_CODER_WGEMM` | KL |
  |---|---|---|
  | fa2 (default) | on (default) | 1.15e-2 |
  | fa2 | 0 | **1.52e-2** |
  | splitk | on | 2.04e-3 |
  | splitk | 0 | 0 (bitwise) |
  | fa2 | gu only | 4.73e-3 |
  | fa2 | down only | 9.70e-3 |

  The FA2 attention (`cx_attn_fa2` + merge) is the stage that diverges at short context; the
  tensor-core GEMMs add ~2e-3 on their own. At 3000 / 8000 tokens both are under 1e-5. Not fixed.
- **`flash-tcheck` panics at `cli.rs:1913`** (index out of bounds on an empty per-layer
  capture) after printing `t1 vs t8: logits differ in 0`, on 4133d39a too. The bitwise logits
  line is the gate; the per-layer table needs a capture the Coder engine doesn't fill.
