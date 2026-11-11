# DeepSeek-V4.1-Flash on tang: the port, milestone by milestone

Status (2026-10-27): M0–M2 landed and **the real model serves**: `tang-llm deepseek-serve` on
1× H200 (RunPod, experts streamed through an 18000-tensor LRU) answers chat and DSML tool calls
correctly through the OpenAI API. Speed is bring-up speed (decode ~1.2 tok/s, prefill
~0.7 tok/s warm): naive kernels, host-side top-k / routing, one GPU. M3–M6 are the speed work. The goal is frog cloud's "bull" model:
`cloud/bull` in frog, served by tang.

Shapes below are from the real `config.json` (`deepseek_v41_text`), not the estimates in
[models.md](models.md), which predates the GLM-5.3 work and is stale on two counts: tang now has
a Coder-Next GPU engine (`flash/coder.rs`) and most of GLM-5.3-Flash on the CPU reference
(`flash/glm5/`). Both cut the DeepSeek bill.

## Speed work landed (2026-10-27, waves 1–2)

Eight parallel lanes, merged; all 30 `flash::deepseek` tests pass on mew (3090) with every
switch at its default. Parity bar unchanged (teacher-forced KL ~1e-15 with `DS_EXACT=1`,
~5e-15 with the fast kernels).

- **Selection on the device** (`kernels_select.cu`): indexer masks, candidate blocks, radix
  top-k (ties to the lowest index), per-expert routing CSR, window index rows. A decode step
  reads nothing back but the logits.
- **Fast kernels** (`kernels.cu`, `DS_EXACT=1` restores the bit-exact ones): tensor-core
  K-split fp8/fp4 GEMVs (61–75% of 3090 bandwidth on the big dense shapes, 26–49% on the
  small ones: launch-floor bound), fused w1/w3 + SwiGLU, tiled 64-head sparse attention
  (10× the exact kernel). Relaxed bar: ≥99.5% bf16-identical, the rest 1 ulp.
- **CUDA graphs** (`graph.rs`, `kernels_graph.cu`, `DS_GRAPH=0` disables): the whole decode
  step and the 6-token verify window captured once per shape; positions come from a device
  params buffer; persistent arena + pinned staging, no lifetime syncs. Prefill, DSpark's
  Markov sampling, the expert LRU and the EP reduce stay outside.
- **Speculative decoding with DSpark** (`decode_spec`, `forward_all`, `snapshot`/`rewind`):
  verify windows at any position, exact rewind (incl. the ratio-2 compressors' partial
  groups, via a journal), greedy and rejection sampling (`SpecPick`: same temperature/top-p
  transform on target and draft; output distribution = plain sampling, checked by TV).
  `serve` uses it at every temperature (`DS_SPEC=0` disables); `Usage` reports drafts,
  accepted, decode steps.
- **Expert parallelism** (`ep.rs`, `tang-moe/src/nccl.rs`, feature `ep`): experts sharded
  by id across GPUs, NCCL all-reduce of the routed partial sum, one thread per device.
  world=1 bit-exact; 2 ranks on one device KL 0.00. **NCCL itself has never run** (needs a
  ≥2-GPU box).

Measured ceiling (kernel lane, real shapes, 10.5 GB/token): at the measured ~52% bandwidth
efficiency ≈ 240 forwards/s on H200, ≈ 400 on B300; with DSpark's 3–4 tokens a forward that
is the 1000 tok/s class, once the launch floor and the all-reduce are hidden.

Not yet: a real-weights run of any of this (only the naive engine has run on the H200);
Engram rows still decode on the host; the max-width indexer launches at long context; the
EP reduce inside the graph.

## Real weights, 4× H200 (2026-10-27, RunPod CA-MTL-3, NVLink, NCCL 2.25)

`tang-llm deepseek-serve /workspace/ds41 --devices 0,1,2,3`, experts sharded across the four
GPUs (89 GB each resident), 35-token code prompt, thinking off:

| mode | tok/s | forwards | tokens/forward | drafts accepted |
|---|---|---|---|---|
| greedy, DSpark spec (200 tokens) | 3.8 | 46 | 4.3 | 154 / 225 (68%) |
| sampled T=0.6 top-p 0.95 (120 tokens) | 3.4 | 31 | 3.9 | 88 / 150 (59%) |
| greedy, no drafts (`DS_SPEC=0`, 100 tokens) | 1.0 | 99 | 1.0 | — |

Same prompt after the wave-3 merge (graphs under EP, grouped MoE) plus the two host-side fixes
below, same 46 forwards and 154/225 drafts, so the math is unchanged:

| build | greedy 200 | sampled T=0.6 200 |
|---|---|---|
| wave 3 merged, as measured first | 1.1 | 0.9 |
| + Engram tables pre-faulted at open | 17.0 | 12.3 |
| + DSpark Markov head on the device | 24.1 | 18.1 |

The 1.1 was not the graphs or the grouped MoE (the A/B sweep of `DS_GRAPH=0` × `DS_MOE_LOOP=1`
disagreed with itself run to run). `deepseek-profile --direct` showed verify steps of 1050 ms
decaying to 133 ms over a dozen steps with 70% of each step outside every op span, and the
decode step's gap sat before `embed`: the step's Engram rows. They are read per step from the
two 100 GB memory maps, ~300 rows of 256 bytes for a 6-token window, and on the pod's MooseFS
mount a cold row is a ~2 ms round trip (measured: 300 random rows 650 ms cold, 5 ms warm). The
first steps of every fresh prompt were the slow ones, which is also why the bench numbers
drifted with page-cache warmth. `Model::open` now touches every page of both tables (1.5 s
warm, ~5 min cold once per boot; `DS_ENGRAM_PREFAULT=0` skips it). The draft step was then
host-bound (120 ms host, 12 ms GPU): the Markov head added each position's bias with a
256 × 129k GEMV on the CPU; it runs on the device now with one row read back per draw.

Wave 4 (three lanes, merged): grouped MoE on the tensor-core K-split path (6.3 → 0.97 ms per
6-token layer on a 3090, 74% of peak; the loop's `ds_linear_csr` is exact so the grouped-vs-loop
test is now a KL/argmax bar, 0.00 on the fixture), `ds_router` block-parallel (656 → 8.7 µs per
layer; it was one serial thread per token recomputing sqrt-softplus six times) with a one-warp-
per-row GEMV for t ≤ 8 raw linears, the NCCL reduce forked onto a side stream so the shared
expert overlaps it, and one launch fewer per mHC mix. Same prompt: 25.1 greedy / 22.8 sampled
(drafts 240/151, the K-split path rounds differently from the exact loop). Less than the lane
benches add up to, so the next step is the per-request table (`DS_SERVE_PROFILE=1`) on the
real model rather than more guessing.

Also in this round: the Engram tables are copied into anonymous host memory at open
(`St::pin`, one shared copy per process, ~200 GB, 7–10 min cold on the pod's network mount;
the page cache that pre-faulting relied on was evicted within the hour), `DS41_CTX`/`--ctx`
now reach the server's limit (it reported 4096 whatever was asked), prefill runs in
`DS_PREFILL_CHUNK`-token chunks (a 10k-token prompt in one forward ran the device out of
memory), and an EP rank that fails logs and exits instead of leaving its peers spinning in NCCL.

**frog end to end** (`FROG_BASE_URL` at the pod's server through an SSH tunnel, `frog -m
cloud/bull -p "Create hello.py that prints the first 10 Fibonacci numbers, then run it."`,
ctx 32768): correct — set a title, tried Write (blocked under `-p`), wrote the file through a
shell heredoc, ran it, summarized. 18.5 minutes wall: frog's prompt is ~17k tokens and each of
its three turns prefilled it from position 0 (17272, 17483, 17626 prompt tokens; no prefix
reuse) at ~50–60 tok/s. For agent use the prefill side now matters more than decode: prefix
caching across turns (the KV is tiny: window ring + compressed) and a faster prefill (the
indexer scores every chunk position against the whole compressed cache, so long prompts slow
down as they grow).

Per-request tables from the wave-4 server (`DS_SERVE_PROFILE=1`, greedy 200, 4× H200): 42.9
tok/s on the graph path (verify 80 ms, draft 9.1 ms with 7 host syncs, decode-iteration host
3.4 ms), 37.7 on the direct path, whose 6-token verify step (90 ms, 3295 launches) splits as:

| op | ms/step | note |
|---|---|---|
| layer.attn | 53.5 | wo 19.1 (0.48/layer), sparse_attn 18.8 (80 launches), compress 9.8 (indexer 6.6, 48 launches), q 3.0, window_kv 1.8 |
| layer.hc | 16.9 | 123 mixes × 137 µs: launch latency, not math |
| layer.moe | 15.2 | grouped_gu 4.7, down 2.3, shared 3.1, reduce 2.4, router 2.2 |
| head | 1.6 | plus 3.1 MB of logits read back and ~3 ms of host verification per step |

So after wave 4 the step is attention + glue latency, and the launch count (3.3k nodes, each
~2–4 µs of front-end time even in a graph) is a 6–12 ms floor of its own: 1000 tok/s (≈ 4 ms
per iteration) needs the per-layer work fused into a handful of launches, not faster versions
of 80 kernels. Wave 5 lanes: attention block (sparse_attn onto the tiled path — it was running
the serial debug kernel —, wo_a one warp per row, the indexer batched over the window's rows:
1378 → 159, 923 → 381, 254 → 76 µs/layer on a 3090, bit-exact), mHC glue fusion and launch
count, the draft as a captured graph with device-side draws, head on the K-split path with
device-side verification.

Wave 5 merged (attention: sparse_attn onto the tiled path, wo_a one warp per row then K-split
per group, the indexer batched over the window; mHC glue fused to 4–5 launches/layer; the greedy
DSpark draft on the device with one read-back; bf16 head at ~88% of bandwidth and the verify
reading back 24 bytes of argmaxes instead of 3.1 MB of logits; the quadratic stop check fixed):
**57.0 tok/s greedy / 42.4 sampled** on the graph path (from 42.9 / 22.8), same prompt. Direct
verify 90 → 66 ms: attention 53.5 → 23.8 (wo 8.1, compress 5.1, sparse_attn 4.0, q 3.2,
window_kv 2.2), head 1.6 → 0.5, host 3.2 → 0.03 ms. Not improved: `layer.hc` 18.5 ms (the
bit-exact post-mix is load-latency bound: 150 µs per 5120×24 GEMV), `layer.moe` 20.6 (reduce
9.2 of it, which is rank 0 waiting: ranks 1–3 report 43 ms per step against rank 0's 66 and
2323 launches against 3354), and on the graph path the draft takes 20.9 ms with 6.9 syncs
(7.8 ms / 2 syncs on the direct path — the device draft isn't taken after a graph verify).
Prefill is 28–53 tok/s (a 17k-token frog turn is minutes). Wave 6 lanes: draft as a graph +
device sampling, mHC with a relaxed reduction order (DS_EXACT=1 keeps the exact kernels),
MoE grouped kernels tuned for 132 SMs + router/route in one launch + reduce overlap, attention
fusion (one wo launch, q in 2, sparse_attn in 1), and prefill (grouped MoE and GEMM tiles for
t > 8).

## Wave 6 measured, 8× H100 80 GB (2026-10-28, RunPod EUR-NO-2, NVLink NV18, NCCL 2.25)

The CA-MTL-3 H200 host never came back (its GPUs went to another tenant and H200/B200 stock
ran out everywhere), so wave 6 was measured on a fresh 8× H100 pod ($31.92/h, 2 TB host RAM:
Engram pin 160 s, ready in 226 s). Same 35-token prompt, 200 tokens out, `bab5bcfc`:

| mode | tok/s | steps | drafts accepted | graph verify | graph draft |
|---|---|---|---|---|---|
| greedy | **130.7** (×3 runs: 130.4–133.5) | 48 | 152/235 | 26.2 ms | 4.0 ms |
| sampled T=0.6 | 68–70 | 49–50 | 149–153/240–245 | 27.0 ms | 13.2 ms |
| greedy, direct path | 98.9 | 48 | 152/235 | 31.8 ms (1386 launches) | 4.4 ms (151) |
| sampled, direct path | 55.8 | 54 | 147/265 | 32.7 ms | 13.6 ms (152) |

Prefill 455–465 tok/s (was 28–53). The 4× H200 wave-5 figure was 57 greedy / 42 sampled, so
wave 6 plus the extra ranks is 2.3×; the ranks are now balanced (every rank 31.7–32.7 ms per
verify step; the old rank-0 skew was the CA-MTL-3 box). Direct-path verify op table (gpu
32.7 ms): `layer.attn` 12.8 (q 2.8 over 80 launches, sparse_attn 3.5 / 80, wo 3.4, compress
2.7), `layer.moe` 11.8 (router 2.0, route 0.8, grouped_gu 3.5, grouped_down 1.7, reduce 2.3,
shared 2.9), `layer.hc` 4.3 over 123 launches, head 0.56. Draft (greedy) 4.4 ms: named ops
cover 15%, the rest is the 3 MTP layers' glue and the device pick.

Why sampled is half of greedy: (1) `ds_spark_sample` is one 1024-thread block over the 129k
vocab with f64 `exp` per element, ~1.9 ms per draw × 5 draws = +9 ms per draft step; (2) the
sampled verify reads a 129k-f32 logits row back per reached position (one sync each) and runs
`SpecSampler::dist` on the host (f64 softmax + select_nth over the vocab), ~18 ms per step
outside any profiled span. Both are wave-7 lanes (device-side rejection verify; a multi-block
f32 sampler). The greedy path's remaining gap to 1000 tok/s (~4 ms per token-step) is launch
count and per-layer glue: 1386 launches for 26 ms of graph replay on 8 H100s, where the
bytes per rank (experts/8 + dense) put the bandwidth floor near 3 ms.

Wave 7 sample lane (same pod, 200 tokens): sampled T=0.6 now **130.1 / 134.1 tok/s** (46 steps,
154–156/225 accepted) against greedy 126.0 / 132.8 on the same binary; graph draft 4.7 ms gpu
(was 13.2), graph verify 26.3 ms with ~0.05 ms host. `ds_spark_sample` buckets w by exponent in
one pass and sorts only the smallest bucket holding the top-p mass (<= 2048 tokens; the old
key search is the fallback), drawing bit for bit as `SpecSampler`; the verify's rejection walk
runs on the device (`ds_spec_dist` per row + `ds_spec_walk`, one small read-back,
`DS_SPEC_HOST=1` for the host walk), also bit-exact against `verify_rows`.

Wave 7 attn3 lane, first step (cc255a81, 8× H100): q's rope and o's inverse rope fused into the
tiled attention's q load / o store, and the window + top-k index rows in one `dg_win_idx`
launch. Verify step 1386 → 1268 launches, direct gpu 31.58 → 31.51 ms (`layer.attn` 12.75 →
12.59: q 2.76 → 2.59, wo 3.43 → 3.29, sparse_attn 3.51 → 3.68), greedy 125.2/133.5 → 125.6/133.3
tok/s (noise), acceptance 152/235 unchanged. Launch count alone buys little here: the attention
kernels are latency-bound in themselves (~45 µs a sparse_attn span for ~8 MB of keys).
`DS_ATTN_FUSE=1` (soft+pv in one launch) measured no better (3.84 ms) and stays opt-in.

Wave 7a merged (sample + attn3 first step, 15a7af70) on the **1017 weights** (swapped in
2026-10-28 20:18 UTC; the Oct 16 `chat_template.jinja` copied over), 8× H100, 200 tokens:
greedy **134.9 / 134.2** tok/s (47 steps, 152/230), sampled T=0.6 **131.7 / 132.0** (45–46
steps, 156–157/220–225), sampled T=1.0 top-p 1.0 (DeepSeek's recommendation) 122.2 (49 steps,
152/240), direct path greedy 101.6; prefill 460–475 tok/s. Full GPU suite green on mew (19
engine/EP/forward + 17 kernel tests).

Wave 7b merged (+ attn3 split-K tensor-core attention and unrolled Sinkhorn, 3c44a203), 1017
weights, 8× H100, 200 tokens: greedy **147.0 / 147.0** tok/s (46 steps, 158/225), sampled T=0.6
141.3 / **152.7** (43–45 steps, 157–158/210–220), sampled T=1.0 145.8, direct greedy 109.6,
prefill 460–470.

Wave 7c merged (+ moe3: shared expert folded into the grouped launches, route+plan+quant in
one launch, `MOE_KS_ITEMS` 4 → 2; 4854062f), 1017 weights, 8× H100, 200 tokens: greedy
**147.3 / 147.2** (48 steps, 153/235; 28.3 ms per step vs 29.6 at 7b — the step got faster, the
run took two more steps), sampled T=0.6 **148.4 / 150.2**, T=1.0 146.1, direct greedy 109.7.

Wave 7d merged (+ attn3 03076460: index rows built inside `ds_attn_fdm`, one launch per layer
gone; d1ad44b0), 1017 weights, 8× H100, 200 tokens: greedy **149.2 / 149.1** (48 steps, 153/235;
27.9 ms per step), sampled T=0.6 148.1 / **152.1**, T=1.0 172.4 (39 steps, 165/190 — an
acceptance-lucky run), direct greedy 112.3. Per step the verify+draft is now ~28 ms; the
remaining budget split is roughly attn 10.6, moe 10.5, hc 3.7 ms on the direct path.

Wave 7e merged (+ attn3 f0d801c9 o-quant in the attention merge, 23213c65 hc stage 21.9 → 16.1
µs per launch; 2eb3da81), 1017 weights, 8× H100, 200 tokens: greedy **154.2 / 152.9** (48 steps,
154/235; 27.0 ms per step), sampled T=0.6 145.9 / **153.5**, T=1.0 137.6 (50 steps), direct
greedy 114.8. The moe3 lane's second round (h quantised in the down kernel, relaxed-order router
GEMV, persistent gu+act+down) measured no gain and was dropped; `layer.moe` stays 10.6 ms,
latency-bound per expert slot (gu 63 µs for ~12 MB of fp4 per layer ≈ 190 GB/s) — the grouped
kernels' grids are the next target, not more fusion.
Wave 8 spec lane (draft length, 1017 weights, 8× H100, 200 tokens, one server per setting;
tok/s is the second greedy run, the first carries captures): `DS_DRAFT_N` sets the drafts a
verify window takes. Below the trained block of 5 the drafter still runs 5 and the window keeps
a prefix; above it the drafter's block grows (more noise tokens than trained), which DSpark
tolerates: the 6th draft is still accepted often.

| n | greedy tok/s | steps | accepted | tokens/step | sampled 0.6 tok/s (2 runs) |
|---|---|---|---|---|---|
| 4 | 136.0 | 53 | 146/208 | 3.77 | 136.7 / 125.2 |
| 5 | 148.9 | 48 | 153/235 | 4.17 | 139.0 / 155.2 |
| **6** | **158.0 / 157.0** | 41 | 163/240 | 4.88 | 82.7 (captures) / 161.7, 132.8 / 139.4 |
| 7 | 151.2 | 41 | 159/280 | 4.88 | 128.1 / 142.3 |
| 8 | 96.5 | 40 | 161/312 | 5.00 | 89.5 / 94.1 |

The step is not flat in t: ~28.0 ms at t 6, 30.9 at t 7, 32.3 at t 8 (≈1.4 ms a row), and
t 9 leaves the 8-row K-split kernels for the prefill path (+20 ms). So serve defaults to
n = 6 (`SERVE_DRAFT_N`); n = 7 wins nothing, its 7th draft rarely lands. `DS_DRAFT_CONF`
(keep the prefix whose product of sigmoid(confidence) stays above it) measured no gain: n 6
c 0.3 156.5 (45 steps, 156/192), n 7 c 0.3 156.1, n 7 c 0.6 141.4 (it cuts drafts that would
have landed). Served default (n 6, b8b2fc79, `measure.sh`): greedy 148.5 / 155.4 / 156.5
(41 steps, 163/240), sampled T=0.6 132.3 / 140.7 (44–45 steps, 156–158/258–264), T=1.0 134.2,
direct greedy 121.3 (7d: 112.3). Greedy gains ~5%; sampled at n 6 over six runs averages ~141
against n 5's 139 / 155 here and 148 / 152 at 7d, so sampled may prefer n 5 (its 6th draft
lands less): a per-request n (greedy 6, sampled 5) is the follow-up. n = 7 first failed to launch: the shared-folded grouped MoE kernels
(`ds_moe_ks_*_s*`) take 46 KB dynamic smem at t 8 over d 5120 and were not opted into > 48 KB.

Wave 8a merged (+ spec lane draft n = 6 default, attn3 99e40728 bf16 split-K GEMV for the
compressor; 850b5e9e), 1017 weights, 8× H100, 200 tokens: greedy **155.7 / 155.8** (43 steps,
162/252, 4.65 tokens/step), sampled T=0.6 148.0 / 143.0, T=1.0 133.4, direct greedy 120.1.

Wave 8b merged (+ spec per-mode draft n: greedy 6, sampled 5; mega lane's shared mHC device
function, DS_MEGA/DS_PDL still off; attn3 ds_ks2 wq_a+wkv in one launch and 4 groups in flight
per warp, bit-exact; d14b36ea), 1017 weights, 8× H100, 200 tokens: greedy **160.1 / 160.4**
(43 steps, 162/252, identical acceptance to 8a), sampled T=0.6 135.0 / 155.2, T=1.0 155.7,
direct greedy 122.4. The moe3 lane's grouped-kernel occupancy work measured 176 greedy on its
own branch and is being merged next.

Wave 8d merged (+ moe3: grouped K-split MoE at 4 blocks/SM with 4 groups in flight per warp,
parallel routing plan, router gate on `ds_gemv_bf16_r1`; one shared `UG` mechanism with attn3's
ds_ks; 6e597f68), 1017 weights, 8× H100, 200 tokens: greedy **177.8 / 179.0** (42 steps,
159/246), sampled T=0.6 153.5 / 175.3, T=1.0 177.4, direct greedy 133.9, prefill ~455.
Mew: `engine::tests::gpu_tiny` 14/0 and `gpu::` 21/0 on 47586ccb (6e597f68 adds docs only).

Wave 8e merged (+ attn3 host fixes: table-driven Engram row decode, bit-exact, `inputs()` 1.35 →
0.13–0.88 ms; greedy decode skips the 3 MB draft-logits readback; 77f95b9f), 1017 weights, 8×
H100, 200 tokens: greedy **192.2 / 193.0** (42 steps, 159/246, identical acceptance), prefill
~507 (Engram gather), direct greedy 141.8. Sampled T=0.6 166.4 / 171.8 in one run vs 182.2 /
188.5 on the base build: being re-measured interleaved before it is called a regression. Of the
~3.4 ms between graphs, verify → draft was 0.83 ms (readbacks, ~60 host-launched rewind copies,
spark_setup staging) and draft → verify 2.24 ms (draft-pack readback, snapshot copies, `inputs()`).

## Status at pause (2026-10-29)

Paused with the 8× H100 pod (bs81p9ixg3coxz, EUR-NO-2) EXITED: the RunPod balance ran out
(restart returns 402). `/workspace` keeps the 1017 weights, the Engram metadata, `measure.sh`,
`bench.sh` and `LANES.md`; `/root` (binaries, build dirs, lane scripts) will be gone on restart.

Measured progression (greedy, 200 tokens, same prompt): 57 (wave 5, 4× H200) → 131 (wave 6,
8× H100) → 154 (7e) → 156 (8a) → 160 (8b) → 179 (8d) → **193 (8e, 70928d3b)**. Sampled T=0.6
moved from 69 to the 150–190 range; sampled runs swing ±20 tok/s with acceptance.

Merged after 70928d3b but **not yet measured on the real model** (all pass the mew GPU tests):
moe3's persistent one-wave expert grid (ee3d0c44, bit-identical, default on); attn3's EP load
deadline (`DS_LOAD_TIMEOUT`), batched snapshot/rewind (`ds_copy_many`), MADV_HUGEPAGE on the
pinned Engram copy; `DS_PIPE=1` device-resident accepted hiddens (off by default); the
`ds_engram_gather` UVA kernel and benchmark (not wired, never run).

Where a verify step goes (nsys node trace, wave 8c/8d): ~20 ms of kernels in the verify graph
with <0.5 ms of gaps, ~4 ms draft, ~3.4 ms between graphs before the 8e host fixes (~1–1.5 ms
after). Largest kernel rows: mHC `ds_hc_fused_ks` 2.9 ms, `ds_ks` 2.8, `ds_attn_fdm` 2.5,
expert gu 2.6 + down 1.7, NCCL all-reduce 1.8 on rank 0 and 3.2–3.9 ms on ranks 1–7.

Open work, in value order:
1. **Rank-0 skew.** Rank 0 spends 4.2 ms/step in expert kernels vs 2.3–2.6 on ranks 1–7, and
   the others wait for it at every layer's all-reduce. Likely cause: the folded shared expert
   runs on rank 0 only. WIP on `lane/moe3` (7ccb7670, unfinished, EP tests were being
   bisected): spread it over all ranks (bit-exact form: every rank does shared gu, down only
   for its 1/8 of output columns, zeros elsewhere).
2. **Expert kernels at 30–44% of HBM.** Next lever: warps covering 2–4 row tiles per mma group
   so the activation decode is shared; bench with real routing (`DS_ROUTE_DUMP`,
   `mega_moe_slots` with `DS_ROUTE_FILE`).
3. **Big fp8 projections at 31–51% of HBM** (wq_b, wo_b, wo_a). Repack for 16-byte loads was
   bit-exact but no faster (parked on `lane/attn3-repack`); the limit looks like per-item ramp
   and tail.
4. **Between graphs.** Measure `DS_PIPE=1` and the UVA Engram gather; full device pipelining is
   a multi-day refactor worth ~4–6% and is deferred.
Measured as no gain and left off: the `mg_glue` megakernel (`DS_MEGA`), programmatic dependent
launch (`DS_PDL`), NCCL protocol/algorithm/channel settings, draft lengths above 6.

To resume: add RunPod funds, start the pod, rsync `frog/deepseek-v41` to /workspace/tang-obs,
`cargo build --release -p tang-llm --features ep,otel --target-dir /root/target-obs`, then
`flock /workspace/gpu.lock /workspace/measure.sh <tag> <bin>`; measure the unmeasured merges
above first. On mew, build with `--no-run` outside `~/tang-gpu.lock` and lock only GPU filters.

Newer weights: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (HF 1bae0e10, 2026-10-17) is a
continued-training checkpoint of the snapshot on the pod (314113f8, 2026-10-16): config.json
byte-identical, same 96,085 tensors in 48 shards, 510 GB, so tang loads it unmodified and the
speed numbers carry over. DeepSeek reports better reasoning/coding/agentic scores and shorter
outputs; recommended sampling T=1.0, top-p 1.0. Packaging differs: no `chat_template.jinja` (the
formatting lives in `encoding/encoding.py` with test vectors under `encoding/tests/`, worth
using as conformance vectors for the Rust formatter). Not yet on the pod: its 700 GB volume
holds the Oct 16 weights (476 GB) and cannot hold both.
Wave 7 attn3, kernel speed (5e153ea2, df2b224a, 8× H100): split-K decode attention
(`ds_attn_fd` → `ds_attn_fdm` on bf16 mma, one block per row × 64 heads × key split, online
softmax, `ds_attn_fd_comb` merge + o rope; `DS_ATTN_FD=0` / `DS_ATTN_FDM=0` fall back) and
`sinkhorn_warp` unrolled for hc = 4 (bit for bit). Bench t 6 k 646: attention 85 → 34 µs; hc
glue 57.6 → 48.0 µs/layer. In the model: greedy 125.2/133.5 → 138.9/148.5 tok/s, verify direct
31.6 → 29.3 ms, `sparse_attn` 3.51 → 1.99 ms, `layer.hc` 4.33 → 3.76 ms, acceptance 158/225
(text differs from the 152/235 run). Still 25 µs per sparse_attn span: the per-block Q load
(128 KB) and the split partials (t × S × h × d) now dominate, not the math.

Wave 6 as merged (895f89b0 → bab5bcfc): attention fusion (one
grouped K-split launch for wo, 19 → 2 launches, 143 → 54 µs on a 3090; q + window_kv fused,
the compressor/indexer key side batched over the window, ~150 → ~30 attention launches per
compressing layer; every fast linear drops its copy + qdq launches), MoE (redundant quant
passes gone, 7 fewer launches per layer; `DS_MOE_KS_ITEMS`/`DS_MOE_KS_S` grid knobs and the
opt-in fused router `DS_MOE_GATE=1` to sweep on H200), the draft as one captured graph with a
single read-back and device-side sampling that reproduces the host's draws bit for bit (66
replay nodes vs 248 launches; an obs counter bug fixed), and prefill: tiled grouped MoE and a
K-split path for raw linears at t > 8, the tiled attention chunked over rows (tiny fixture
120-token prefill 23 → 13 ms, 3696 → 592 launches; bench at real shapes: wq_b 3.7 → 1.4 ms,
wo_b 4.7 → 2.3 ms at t = 512). The mHC lane landed last (852e2eb6): the mixes' GEMV on the K-split path
with the post-mix folded in and the whole stage in one launch behind a grid barrier — 2 glue
launches per layer at t ≤ 8, 155 → ~50 µs/layer on the 3090; `DS_HC_EXACT=1` (or `DS_EXACT=1`)
keeps the bit-exact kernels, which `gpu_tiny_free_running` now pins (the tiny random fixture
amplifies ulp drift over free-running steps: 6/10 top-1 relaxed, 10/10 exact; the relaxed path
is held by `gpu_tiny_hc_ks_matches_exact`, KL 0 and argmax unchanged teacher-forced). The real
model's draft acceptance rate is the check that matters for that choice. Full GPU suite on
mew at b178b8a2: 12 engine, 2 EP, 14 kernel tests. Still open: the per-rank table (rank 0 was
66 ms vs 43 on the others); a real multi-token GEMM tile for prefill.

Steady state after the host fixes: verify ≈ 130 ms and draft ≈ 15 ms for 4.3 tokens, against a
~4 ms floor. The verify step's remaining cost is the grouped MoE on the non-tensor-core `lin_fast`
path (`moe.router` 28 ms and `moe.reduce` 10 ms per step are the next two line items), then
the launch count (3.3k per verify step).

Output is correct (a working CSV parser with tests). NCCL expert parallelism matched the
single-GPU engine at KL 0.00 on the tiny fixture at 2 and 4 ranks before this run.

So tokens/forward is where it needs to be (DSpark gives 3.8× over plain decoding, and a
6-token verify window costs about the same as a single step); the forward is ~1.0 s against a
~4 ms bandwidth floor. Two causes, both being fixed: (1) `graphable()` is false under EP (the NCCL reduce is
a host callback), so EP steps run the direct path with its syncs; (2) the MoE loop launches
~5 kernels for each of the 384 experts per layer (~80k launches a forward) instead of one
grouped launch over the routing CSR.

## Wave 8 mega lane: the per-step floor and the megakernel question (2026-10-28, 8× H100)

**Launch floor** (`mega::tests::mega_launch_floor`, one H100, 132 SMs, CUDA 12.8): a captured
graph of N dependent trivial kernels, µs per node, flat in N (300–2400):

| chain | µs / node |
|---|---|
| graph, 1-block kernels | 0.91 |
| graph, 132-block kernels | 1.10 |
| graph, 132-block, a memcpy node every 10 | 1.08 |
| plain stream launches, 1 block | 1.81 |
| plain stream launches, 132 blocks | 1.99 |
| grid barrier inside one persistent kernel, 132 × 256 (ds_hc_fused_ks's) | 2.08 |
| grid barrier, 132 × 1024 / 264 × 256 / 64 × 256 / 8 × 256 | 2.10 / 2.71 / 1.97 / 1.69 |

So the verify graph's ~1188 nodes cost **~1.3 ms** of its 23 ms replay as pure node overhead.
The rest is ~19 µs per node of kernel bodies, mostly latency-bound: memory round trips,
serial phases, and tails on kernels that move a few MB. **A grid barrier costs as much as two
graph nodes.** A persistent kernel saves nothing on synchronization; it only wins where it
overlaps work across what used to be kernel boundaries.

**Per-layer launch inventory** (nsys on the tiny fixture, verify window t = 6, direct path,
254 kernels over 8 layers + head, ~30 per layer; the real model at 948 direct launches per
verify is ~24 per layer): mHC 2 (`ds_hc_fused_ks` attn + ffn, plus `ds_mean_hc` on DSpark
target layers); attention prologue 5 + 1 memcpy (`ds_ks_quant`, wq_a and wkv `ds_ks`, ring copy,
`ds_qkv_post`, wq_b `ds_ks`); compressor/indexer 3–8 on compressing layers (`ds_linear_t8`,
`dg_rope`, `dg_index_score`, `dg_topk`, ...); attention core 1–2 (`ds_attn_fdm` + comb, or
`dg_win_idx` + sparse attn); wo 2–3 (`ds_wo_ks`, wo_b quant + `ds_ks`); MoE 5 (`ds_router`,
`ds_route_plan_q`, grouped gu, grouped down, `ds_moe_reduce_sh` + the NCCL all-reduce).

**The dense-glue megakernel** (`mg_glue` in kernels_mega.cu, `DS_MEGA=1`, e93df959 + 3e4e8677): the 6
launches + 1 memcpy from the MoE all-reduce of layer l−1 to the attention core of layer l
(the attention mHC stage, the block input's quantization folded into its store, wq_a, wkv, the
ring copy, `ds_qkv_post` with its 1024-thread reductions replayed in order by 256 threads,
wq_b) as one persistent launch with 3 grid barriers (+ one among the mHC blocks), idle blocks
prefetching the projections' weights into L2. **Bit for bit** the separate launches
(`gpu_tiny_mega_matches` on mew and H100; `mega_glue_bench` at real shapes). Measured:

| | separate | mg_glue |
|---|---|---|
| bench, real shapes, t = 1 / t = 6, µs per layer (graph of 40, weights rotated past L2) | 89.6 / 103.0 | 105.1 / 117.9 (2 blocks/SM; 1/SM: 126 / 137; L2 prefetch 0–40 MB: ±2) |
| server, direct verify: launches | 948 | 744 |
| server, direct op table: glue stretch per layer | ~119 µs (hc attn stage + q + window_kv) | 126 µs (`layer.attn.glue`) |
| server, graph verify / draft gpu ms | 22.98 / 3.89 | 23.58 / 3.90 |
| greedy 200 tok/s (×2 warm) / acceptance | 148.4 / 153 of 235 | 143.9 / 153 of 235 |

So it is **slower by ~15 µs per layer and stays off**. Why: the separate `ds_ks` launches
each size their grid from their own occupancy (3–4 blocks per SM for the small wq_a/wkv
staging), the megakernel must keep every block co-resident for its barriers and so runs all
phases at the worst phase's occupancy (2 per SM, set by wq_a's 35 KB activation staging and
the mHC phase's static arrays); its 3 barriers (~2 µs each) replace 6 nodes (~1.1 µs each); and
L2 prefetch buys nothing because the phases are latency-bound, not bandwidth-bound (wq_b's
42 MB at 3.35 TB/s is 12.5 µs of the ~100).

**Plan for the next wave.** The step is ~23 ms of graph against a ~3 ms HBM floor and a ~1.3 ms
node floor; launch count is worth ≤ 1.3 ms in total, so neither a full per-layer megakernel
nor fewer launches alone gets near 4.3 ms. The time is in latency-bound kernel bodies: the
glue stretch moves ~51 MB a layer (15 µs at HBM rate) in ~100 µs; MoE ~264 µs a layer for
~1/8 of 6 × 3 fp4 experts + the shared one.

- *Full per-layer persistent megakernel* (one launch per layer, ~10 phases, ~10 barriers):
  saves ≤ 24 nodes × 1.1 µs − 10 × 2.1 µs ≈ 5 µs a layer, 0.2 ms a step, before the
  occupancy penalty measured above, and the NCCL all-reduce cannot live inside it (it needs a
  device-initiated reduce: NVSHMEM or a hand-rolled NVLink P2P reduce over mapped peer
  memory, itself a wave). Estimated step: 22–23 ms. Not worth scheduling as is.
- *Few big kernels per layer* (≈ 4: mHC+prologue GEMVs, attention+wo, router+route+grouped
  MoE, reduce+post-mix), each a **stream-K / split-K GEMV that keeps every SM busy and
  overlaps the next weight stream with the current reduction** (the real win of fusion, which
  needs per-phase occupancy, not one co-resident grid): the target is each kernel at ≥ 60% of
  HBM rate. At that rate a layer's ~55 MB dense + ~8 MB expert bytes per rank is ~30 µs, a step
  ~1.5–3 ms of bandwidth + ~0.2 ms of nodes + the all-reduces (40 × ~30 µs = 1.2 ms today on
  NVLink; overlap with the shared expert already exists). Estimated step ~5–6 ms, the
  realistic road to 1000 tok/s. Concretely: (1) make `ds_ks` / `lin_mma` bandwidth-bound at
  t ≤ 8 (measure each projection's achieved GB/s with `mega_glue_bench`'s harness; wq_b is
  42 MB in ~25 µs today); (2) chain kernels with programmatic dependent launch
  (`cudaLaunchAttributeProgrammaticStreamSerialization`, graph-capturable) so a kernel's weight
  loads start under its predecessor's tail, which is the overlap a megakernel would buy
  without its occupancy cost; (3) only then fuse the tiny ones (`ds_ks_quant` into its
  producer, as `mg_glue` does bit for bit).

Code: `flash/deepseek/mega.rs` (benches, `Mega`), `kernels_mega.cu`, `mega_engine.rs` (the
deferral hooks in `block_ks` / `qkv_fused`; flushes before any other reader of the stage).

### Wave 8b: per-projection bandwidth and programmatic dependent launch (8× H100 pod, one GPU)

**Bandwidth targets** (`mega::tests::mega_bw_table`, t = 6, real shapes, random weights,
each through the launcher the engine uses, copies rotated past L2, graph replay; % of
3.35 TB/s):

| projection | rows × cols | MB | µs | GB/s | % HBM |
|---|---|---|---|---|---|
| wq_a (fp8, `ds_ks`) | 1280 × 5120 | 6.3 | 14.1 | 466 | 14% |
| wkv (fp8, `ds_ks`) | 512 × 5120 | 2.5 | 9.5 | 275 | 8% |
| wq_b (fp8, `ds_ks`) | 32768 × 1280 | 40.0 | 28.9 | 1450 | 43% |
| wo_a (fp8, `ds_wo_ks`, 8 groups) | 8192 × 4096 | 32.0 | 25.6 | 1313 | 39% |
| wo_b (fp8, `ds_ks`) | 5120 × 8192 | 40.0 | 41.5 | 1013 | 30% |
| indexer wq_b (fp8, `ds_ks`) | 4096 × 1280 | 5.0 | 6.7 | 789 | 24% |
| compressor wkv (bf16, `ds_linear_t8`) | 512 × 5120 | 5.0 | 33.3 | 157 | 5% |
| compressor wgate (bf16, `ds_linear_t8`) | 512 × 5120 | 5.0 | 33.3 | 157 | 5% |
| indexer weights_proj (bf16, `ds_linear_t8`) | 32 × 5120 | 0.3 | 30.8 | 11 | 0% |
| router gate (bf16, `ds_linear_t8`) | 384 × 5120 | 3.8 | 33.0 | 119 | 4% |
| hc mixes (f32, `ds_linear_t8`; the engine runs them inside `ds_hc_fused_ks`) | 24 × 20480 | 1.9 | 104.4 | 19 | 1% |
| expert gu, one slot (fp4, `ds_ks` gu; a stand-in for the grouped launch) | 2 × 2304 × 5120 | 12.0 | 22.5 | 557 | 17% |
| expert down, one slot (fp4, `ds_ks`; stand-in) | 5120 × 2304 | 6.0 | 13.9 | 451 | 13% |
| head (bf16, `ds_head_bf16`) | 129280 × 5120 | 1262.5 | 516.9 | 2561 | 76% |

Read: only the head streams well. `ds_linear_t8` has a ~31 µs floor whatever the size (the
router, compressor and indexer weights; ~4 of them a compressing layer, ~2.5 ms a step): move
the bf16 t ≤ 8 linears onto a K-split GEMV (bf16 `lin_mma` variant), the cheapest ms in the
step. The small fp8 projections (wq_a, wkv) sit at 8–14%: they are a fixed ~9 µs of
ramp/reduction; batching wq_a + wkv into one launch (same activation) and the K-split reduce
in fewer rounds are the levers. The big ones (wq_b, wo_a, wo_b) are at 30–43%; target ≥ 70%
(the head's kernel shows the hardware gives it at t ≤ 8).

**Programmatic dependent launch** (`pdl.rs`: `cuLaunchKernelEx` +
`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`; `mg_pdl_step` triggers
`griddepcontrol.launch_dependents` first, streams its own weights, then
`griddepcontrol.wait`s and bumps the counters its predecessor bumped;
`mega::tests::mega_pdl_floor`, chain of 400, 264 × 256 blocks). **The attribute survives graph
capture** and the order holds (every counter exactly the expected count):

| per-kernel weights | graph plain µs | graph PDL µs | stream plain | stream PDL | bytes at 3.35 TB/s |
|---|---|---|---|---|---|
| 0 | 1.19 | 0.82 | 2.05 | 1.43 | 0 |
| 256 KB | 1.80 | 0.83 | 2.79 | 1.42 | 0.08 |
| 1 MB | 2.40 | 0.86 | 3.57 | 1.43 | 0.31 |
| 4 MB | 3.85 | 1.36 | 5.20 | 1.69 | 1.25 |
| 16 MB | 9.65 | 5.26 | 11.01 | 5.27 | 5.0 |

So PDL takes the node cost from 1.2 to 0.8 µs and, more usefully, **hides a kernel's
independent weight stream under its predecessor** (4 MB: 3.85 → 1.36 µs, at the bandwidth
line). At ~24 kernels a layer that is ≥ 0.4 µs × 960 ≈ 0.4 ms a step from the nodes alone, and
up to each kernel's ramp (the ~9 µs floor of the small projections above) where the weight
loads can move before the wait.

Not wired into the engine yet: a PDL launch is only safe for a kernel that
`griddepcontrol.wait`s before reading its predecessor's outputs, and the dense stretch's
kernels (`ds_ks`/`lin_mma`, `ds_qkv_post`, `ds_hc_fused_ks`, `ds_wo_ks`) live in kernels.cu,
owned by the kernel lanes. The change per kernel is two lines (`mg_pdl_trigger()` at entry,
`mg_pdl_wait()` after the weight-only prologue; in `lin_mma` that is after the activation
staging would move below the first weight loads, i.e. issue the first `KS_UG` groups' weight
loads, wait, then stage x), plus launching through `pdl::launch` when `DS_PDL=1`. Scheduling it
with the kernel lanes is the next step; the expected win on the dense stretch is ~1–2 µs per
kernel boundary (~1 ms a step), more where a prologue is restructured to load weights first.

**DS_PDL=1 wired** (0d155a69): every kernel in kernels.cu / kernels_select.cu /
kernels_graph.cu / kernels_mega.cu opens with `DS_PDL_ENTRY` (launch_dependents, then wait);
`ds_ks` triggers, L2-prefetches its first work item's weight slice, then waits; `obs::launch`
(every deepseek launch) uses `cuLaunchKernelEx` with the attribute when `DS_PDL=1`. NCCL,
memcpy and event edges are untouched (a PDL launch after a non-kernel node is an ordinary
dependency). Full tiny GPU suite with `DS_PDL=1` on H100: 41 passed, 0 failed (bit-exact
tests included; graph capture keeps the attribute). Step, 8× H100, same binary:

| | DS_PDL off | DS_PDL=1 |
|---|---|---|
| graph verify gpu ms | 23.10 | 23.24 |
| graph draft gpu ms | 3.92 | 3.90 |
| direct verify gpu ms (948 launches) | 27.77 | 27.25 |
| greedy tok/s (warm ×2) / acceptance | 147.1, 147.7 / 153 of 235 | 146.3, 144.8 / 153 of 235 |
| sampled 0.6 tok/s | 127.6, 141.7 | 149.5, 137.5 |
| prefill tok/s | ~460 | ~415 |

So **no step gain in the graph** (+0.14 ms verify, within noise to slightly worse), −0.5 ms on
the direct path (where it hides stream launch gaps), and prefill 10% slower (early-launched
dependents holding SM resources next to big tiled kernels). Inside a graph the nodes are
already back to back, and the real kernels' prologues are not weight-only (most load their
activation first), so there is little to overlap with a wait at entry. Stays off. What would
make PDL pay is per-kernel restructuring (weights first, activation after the wait) inside
the big projections, which belongs with the kernel lanes' bandwidth work above.

### Wave 8c: where the graph's verify step goes (nsys node trace, 8× H100, b83fde7d)

`nsys launch -t cuda --cuda-graph-trace=node` around `deepseek-serve` (DS_PDL off, greedy,
draft n = 6, so verify windows are t = 7), one 40-token request captured with `nsys start`/`stop`
after warm-up: /workspace/attn3-prof.sh, report /workspace/prof-attn3.nsys-rep, analysis
/workspace/attn3-an.py and attn3-an2.py (kernels grouped by the graph launch's correlation id,
device 0, the 9 verify replays of 908 nodes; ranks look alike).

- Verify graph: **22.43 ms** span, 21.97 ms of kernels, **0.45 ms of idle gaps** between its 908
  nodes. The draft graph (129 nodes) adds 4.2 ms with 0.07 ms of gaps.
- Step period (verify start to next verify start): **30.0 ms**, of which device 0 runs kernels
  26.6 ms. So ~3.4 ms a step sits **between** graphs (host: acceptance, sampling, the next
  draft's setup), more than all in-graph gaps together.
- wo_b **is** on `ds_ks` in serve: the `ds_ks_f2_g0` with 64,736 B smem (t = 7 × 8192 columns)
  right after `ds_wo_ks_f2` (wo_a) and its `ds_ks_quant`, 38.1 µs (40 MB: 1.05 TB/s, 31%).
  wq_b is the 10,304 B one (1280 columns), 25.6 µs.

Top kernels of one verify step (device 0, mean of 9 replays; 40 layers, so 40 launches is one a layer):

| kernel (what) | launches | µs each | ms / step | % of kernel time |
|---|---|---|---|---|
| `ds_hc_fused_ks` (mHC pre/post, two a layer) | 80 | 36.2 | 2.90 | 13.1% |
| `ds_moe_ks_gu_f3_s2` (expert gate/up) | 40 | 63.3 | 2.53 | 11.4% |
| `ds_attn_fdm` (split-K attention) | 38 | 64.6 | 2.46 | 11.1% |
| `ds_route_plan_q` (routing plan) | 40 | 41.4 | 1.66 | 7.5% |
| `ds_moe_ks_down_f3_s2` (expert down) | 40 | 40.6 | 1.62 | 7.3% |
| `ds_linear_t8` 48 blocks (router gate, bf16 384 × 5120) | 40 | 39.7 | 1.59 | 7.2% |
| `ds_ks_f2_g0` wo_b (5120 × 8192) | 40 | 38.1 | 1.53 | 6.9% |
| NCCL AllReduce (f32 ring LL) | 40 | 35.1 | 1.40 | 6.3% |
| `ds_wo_ks_f2` (wo_a) | 40 | 26.9 | 1.08 | 4.8% |
| `ds_ks_f2_g0` wq_b (32768 × 1280) | 40 | 25.6 | 1.03 | 4.6% |
| `ds_ks2_f2` (wq_a + wkv) | 40 | 17.4 | 0.69 | 3.1% |
| `ds_head_bf16` (vocab head) | 1 | 578.3 | 0.58 | 2.6% |
| `ds_linear_t8` 4 blocks (compressor / indexer bf16) | 8 | 51.8 | 0.42 | 1.9% |
| `ds_router` | 40 | 8.8 | 0.35 | 1.6% |
| `ds_attn_fd_comb` (split-K combine) | 40 | 7.4 | 0.30 | 1.3% |
| everything else | | | 2.04 | 9.2% |
| **total** | 908 | | **22.18** | |

Read: the step is kernels, not launch gaps (0.45 ms in the graph). Besides the projections,
the biggest non-GEMV items are mHC (2.9 ms), the routing plan (1.7 ms, which is longer than the
router GEMV it follows) and the bf16 router gate on `ds_linear_t8` (1.6 ms at 39.7 µs for
3.8 MB, the ~31 µs floor of Wave 8b). Outside the graph, the ~3.4 ms between graphs is the
largest single lever.

## How to profile a step

The step timeline (`flash/deepseek/obs.rs`, [observability.md](observability.md#deepseek-v41-flash-the-step-timeline-flashdeepseekobsrs))
says where a forward's time goes; no guessing.

```bash
# the table: per-forward wall/gpu/host, launches, syncs, bytes; per-op means with % of step
tang-llm deepseek-profile /workspace/ds41 --devices 0,1,2,3 --steps 20 --graph-debug
# the same with decode/verify interiors timed (graphs off for the run)
tang-llm deepseek-profile /workspace/ds41 --devices 0,1,2,3 --steps 20 --direct
# while serving: a line per forward, JSON lines for later
DS_OBS=1 DS_OBS_FILE=/tmp/steps.jsonl tang-llm deepseek-serve /workspace/ds41 --devices 0,1,2,3
```

Read it as: `gpu_ms` vs `wall_ms` (host-bound if they differ), `launches` per forward (a
graph replay is 0 on top of its inputs), `path` (did the decode step actually replay a graph,
see `--graph-debug`), `syncs` (1 on the token path), `expup` (expert uploads after warm-up
mean the LRU or a table rebuild), and per-op `%step` plus `reduce_wait` per rank for EP.

Overhead, RTX 3090, tiny fixture, decode µs/step (best of 10 rounds of ~100 steps):

| | `DS_OBS` unset | `DS_OBS=1` | `DS_OBS=2` (every step) |
|---|---|---|---|
| graph path | 2151 | 2157 (+0.3%) | 2157 (+0.3%; graph interiors aren't recorded) |
| direct path (`DS_OBS_DIRECT=1`) | 2527 | — | 3304 (+31%: ~1100 events on a 554-launch step) |

So `DS_OBS=1` is free; `DS_OBS=2` costs only on the sampled (`DS_OBS_EVERY`, default 50)
direct steps it records, and is for profiling runs. On the tiny decode step the interior's ops
cover 94–99% of the step's device time.

## M0 findings (2026-10-27)

Read from the checkpoint's `config.json`, `model.safetensors.index.json`, shard headers (range
requests) and DeepSeek's reference `inference/model.py`, `engram.py`, `kernel.py`, `convert.py`.

- **Experts are MXFP4 in all but name**: e2m1 values packed two to an `I8` byte along K, low
  nibble first, one `F8_E8M0` scale per row × 32. tang's MXFP4 GEMV (`tang-compute` moe) is the
  starting point, not a new format. 40 × 384 × 3 × 2304 × 5120 / 2 bytes = **272 GB**.
- **Dense weights** are `F8_E4M3` with `F8_E8M0` (power-of-two) scales per 32 × 32 block.
  `wo_a` is dequantized to bf16 at load (convert.py does the same). Norms, `embed`, `head`,
  the router, compressor and indexer projections are bf16; mHC and sinks are f32.
- **No dense layers.** All 40 layers route; 6 of 384 plus one shared expert.
- **Every GEMM quantizes its activation** to fp8 per 32 with power-of-two scales first. The
  window KV and the attention output are fp8 quant-dequantized; compressed KV and index keys
  fp4 (block 16 with e4m3 scales, and block 32 with e8m0, respectively).
- **Engram tables are `F8_E4M3` [384M, 256] + `F8_E8M0` [384M, 8]: ~101 GB each, ~202 GB
  for both.** Each token reads 3 × 8 = 24 rows a layer (256 B each). Host RAM or NVMe with a
  row cache, as PLE.
- **Attention** (model.py `Attention`): every layer attends over a 128-token ring of raw KV
  plus, when `compress_ratio > 0`, `index_topk` = 512 compressed positions. Only
  `kv_source_layers` compress (ratio 2: softmax-gated pooling of 2 tokens; ratio 1: a plain
  projection); the layers after a source read its cache. Only `index_source_layers` run the
  indexer; the others reuse the last source's top-k. Layer 20 also picks 2048 candidate blocks
  of 8 that later indexers are confined to. Ratio > 0 layers use YaRN at θ 160000 for all
  their RoPE; window-only layers plain θ 10000. A learned per-head sink logit joins the
  softmax denominator, and the output has the query's RoPE removed again.
- **mHC** carries 4 copies; each sublayer's Sinkhorn mixes are applied by the *next* sublayer.
- **DSpark** is the 3 `mtp.*` blocks: a block of 5 (the sampled token + 4 noise tokens)
  attends over its own window KV, seeded from target layers 37–39's inputs, plus the whole
  block (unmasked within it). A rank-256 Markov head adds a bigram bias to each draft
  position as it samples left to right; a confidence head scores each draft.
- **safetensors 0.4.5 can't read the headers** (no `F8_E8M0`). The port reads them itself.

Reference and fixture (no DeepSeek code is run): `scripts/deepseek_ref.py` is a torch
transcription of model.py with its kernels' numerics and stable (lowest-index) top-k ties;
`scripts/deepseek_tiny.py` writes a tiny model in the real checkpoint format plus dumps to
`/private/tmp/ds41-tiny`. Prefill-then-decode agrees with one long prefill exactly for 1–3
decode steps; later steps drift because batched and single-row bf16 rounding flips indexer
top-k near-ties (layer 0, which has no indexer, already differs by 3e-4). So parity replays
the fixture's own chunking and is judged by KL and top-1, as for GLM.

## The model, in tang's terms

| Piece | config | tang has | Gap |
|---|---|---|---|
| Residual | 4-stream HC, Sinkhorn, 20 iters, eps 1e-6 | `glm5/mhc.rs` (same shape) | none on CPU; GPU kernel is shared work with GLM |
| Attention | 40 layers, 64 q heads × 512, **1 kv head**, rope 64 of 512, q_lora 1280, o_lora 1024 in 8 groups, sliding window 128, YaRN ×16 over 64K | QSA indexer (Flash-Next), MLA latents (`glm5/mla.rs`) | the 512-wide single-head kernel, the window, o_lora groups, YaRN |
| Compressed attention | `compress_ratios` per layer: 0 / 2 / 1 (layers 0–1 none, 2–19 ratio 2, 20–39 ratio 1), own rope θ 160000 | nothing | **new**: a compressed KV per layer class |
| Cross-layer KV | KV written only at layers 2, 8, 14, 20 (`kv_source_layer_ids`); the rest read a source layer's cache | nothing | **new**: cache owned by a source layer, read by its clients |
| Indexer | 32 heads × 128, top 512 (`index_topk`), index written at layers 2, 8, 14, 20, 24, 28, 32, 36 | QSA's indexer and top-k | sharing across layers |
| Candidate blocks | from layer 20, block 8, top 2048 blocks (`candidate_*`) | glm5 pooled indexer (kpool 4, top 2048) is the closest | **new**: a block-level pre-filter in front of the token indexer |
| Router | 384 experts top-6, `sqrtsoftplus`, `noaux_tc`, scale 1.5, `norm_topk_prob` | `glm5/moe.rs` (sigmoid + noaux_tc, scale 2.5) | the scoring function; one more variant |
| Experts | FF 2304, SwiGLU limit 10, 1 shared, **FP4** weights | Q2_0, Q4X, MXFP4 GEMVs; SwiGLU limit (glm5) | an FP4 expert blob + grouped GEMV; confirm the FP4 format (block scales?) from `quantization_config` |
| Dense weights | FP8, 32×32 block scales, dynamic activations | FP8 safetensors loading (glm5) | block-scaled FP8 GEMV |
| Engram | 2 layers (1, 14), n-grams ≤ 4, 16M hashed vocab, 8 heads × 256, tables of 384,006,168 and 384,016,682 rows | PLE n-gram table with `O_DIRECT` reads and a row cache (Flash-Next) | the hashing, the two-table layout, and **sizing**: the rows alone are ~100 GB a table at bf16, so they live on host/NVMe like PLE does |
| MTP | 3 next-token layers | `flash/mtp.rs` chain (one layer, Qwen) | a 3-layer chain |
| DSpark | drafter: 128 experts top-3, markov rank 256, block 5, noise token 128799, reads target layers 37–39 | verify windows in graphs, `DraftPolicy` | **new**: a second small MoE model as the drafter |
| Vocab | 129,280 | — | tokenizer only |
| Vision | 32-layer encoder, ≤ 1024 image tokens | `qwen3vl.rs` pattern | not in scope for bull |

No `first_k_dense_replace` in the config, so assume all 40 layers route to experts until the
modelling code says otherwise. `compress_ratios` has 43 entries: 40 layers plus the 3 MTP layers.

### Bytes

- Experts: 40 × 384 × 3 × (5120 × 2304) ≈ 544B params. At FP4 ≈ **272 GB**; at 2-bit ≈ 136 GB.
- Active per token: 6 + 1 experts × 40 layers ≈ 9.9B expert params + dense ≈ **~13B** (so the
  "16B" in press copy counts something else; measure it in M0).
- Engram: two tables of ~384M rows. Row width is 8 × 256 at whatever dtype the shards use; read
  the shard index in M0 before believing any number.

The expert bytes decide the hardware. 272 GB does not fit one H200 (141 GB) or one B200
(192 GB) with KV and Engram cache. Either experts stream from host (tang-moe) or they shard
across GPUs. Streaming bounds decode near 6 × 3 × 11.8M × 0.5 B ≈ 35 MB a layer, ~1.4 GB a
token, so ≈ 25–40 tok/s on PCIe 5 before cache hits. The 800 tok/s class needs every expert in
VRAM: **4× B200/B300 or 8× H200, with expert parallelism tang does not have.** That is the
largest item in this plan and the reason the old 6–10 week figure was low.

## Milestones

Each one is a branch with a measured result in the commit body, as tang does now. Weeks are for
one engineer who knows the Flash and glm5 code; M1 and the kernels in M2 can run in parallel.

**M0 — truth (days)**
- Pull `config.json`, the modelling file, `quantization_config` and the safetensors index.
  Write down the FP4 format, the Engram row dtype, and whether any layer is dense.
- Tiny random-init `deepseek_v41` checkpoint via transformers, like `scripts/glm5next_tiny.py`.
  HF fp32 golden logits for a fixed prompt set, per-block dumps for `check_logits.py`.
- Pick the hardware target and price it: 4× B200 vs 8× H200 on RunPod.
- Exit: a sizing table with measured bytes; a tiny fixture under `tests/fixtures/deepseek/`.

**M1 — CPU f32 reference forward (2 weeks)**
- `flash/deepseek/` on the `glm5/` layout: `gguf`/`safetensors` geometry, `forward.rs`, one
  module per owner.
- Reuse: `mhc.rs` as is; `moe.rs` with a `sqrtsoftplus` scorer; glm5's FP8 loader; SwiGLU limit.
- New: the single-kv-head attention with window and o_lora groups; compressed KV with its own
  rope; cross-layer KV and index sharing; candidate blocks; Engram hashing and lookup against
  the real tables (mmap, no cache yet); YaRN.
- MTP layers and DSpark off. Vision off.
- Exit: tiny-model logits match HF fp32 at the existing KL bar, block by block; on the real
  model, the first two layers (no compression, no sharing) match HF on a 512-token prompt in RAM.

**M2 — one-GPU decode with experts offloaded (3 weeks)**
- Engine on the `coder.rs`/`glm5/engine.rs` pattern, CUDA first.
- Kernels: FP4 expert grouped GEMV (sibling of the Q2_0 and MXFP4 ones), block-scaled FP8
  GEMV for dense, the 512-wide attention with window + indexed sparse + compressed reads,
  Sinkhorn mHC on GPU (shared with GLM's speed plan), `sqrtsoftplus` + top-k.
- Engram through the PLE path: `O_DIRECT` row reads, row cache, prefetch off the n-gram of the
  tokens already known.
- Experts in a tang-moe pinned arena with the VRAM cache; proving ground is a forced small
  cache so every number has an all-VRAM twin, as hybrid-moe M5 did.
- Exit: parity with M1 (KL ≤ 0.03 on the prompt set, top-1 flips ≤ 2%); a first tok/s on one
  H200, expected 10–40; `tang-verify` shadow checks on every new kernel.

**M3 — expert parallelism (3–4 weeks; the new infrastructure)**
- Fill the `AllReduce` slot in tang-compute (`docs/extensions.md`) with NCCL through cudarc.
- EP: experts sharded by id across ranks; dispatch and combine as all-to-all; dense weights,
  attention and Engram replicated (they are small). One process, one thread per GPU, one
  CUDA graph per rank per T.
- Rank-0 owns sampling and the KV; the others mirror block ids (the design `ext.rs` already
  assumes).
- Exit: every expert in VRAM on the M0 box; T=1 decode ≥ 100 tok/s batch 1; parity unchanged.

**M4 — drafting (2 weeks)**
- 3-layer MTP chain in `mtp.rs`; `DraftPolicy` picks MTP, suffix or DSpark.
- DSpark: load its 128-expert MoE and Markov table, feed it target layers 37–39, block of 5,
  verify 6 a window. It is a second small engine; reuse the EP dispatch so its experts shard too.
- Exit: `speculate.rs` passes with drafts forced wrong (identical output); ≥ 3.5 tokens a
  window on the code prompt; decode ≥ 2.5× T=1.

**M5 — low-latency serve and bull (2 weeks)**
- CPU between steps: one graph launch per step, DP sync off the hot path, KV admission capped,
  as vLLM's DeepSeek post lists. Measure % of speed of light per step as tang does.
- `serve_glue.rs` for the OpenAI endpoint; `bull` in frog's `STATIC_CLOUD`
  (`crates/frog-engine/src/catalog.rs`), fallback as today.
- Exit: single-stream P90 tok/s published in `speed-log.md` with the SOL fraction;
  `frog -m cloud/bull` runs the frog-bench suite end to end.

**M6 — prefill and long context (ongoing)**
- Prefill kernels for compressed attention and the candidate pre-filter; bounded replay for
  cached prefixes; YaRN past 64K to the 1M the config declares; int8 KV above 8K.
- Exit: 128K prefill tok/s and decode at 128K within 10% of decode at 4K.

Sum: **~13–15 weeks** for one engineer, ~10 with two (one on the reference and formats, one on
kernels and EP from M0). Expert parallelism is a third of it and is the part that also unlocks
GLM-5.3-Flash at speed, so it is not DeepSeek-only work.

## Risks

- **The FP4 format.** If it is a DeepSeek-specific block layout rather than NVFP4/MXFP4, the
  GEMV is new and the requant path (`flash/requant.rs`) needs a target. Settle it in M0.
- **Compressed attention semantics** are only in the modelling code. Ratio 2 vs 1 vs 0 and how
  compressed and raw KV combine decide the kernel; read before designing.
- **Engram size.** Two ~100 GB tables on host means the row cache's hit rate sets prefill speed.
  PLE's numbers are for a far smaller table.
- **EP is new.** NCCL through cudarc, all-to-all in a graph, and multi-rank determinism are all
  first-time code in tang. Budget the slip here, not in the model work.
- **Hopper vs Blackwell.** FP4 tensor cores exist only on Blackwell; batch-1 GEMV dequant is
  fine on H200, but prefill and any batching want B200.
- **Press numbers.** 1500 tok/s is a vendor demo on a Tetris prompt with thinking off. The
  reproducible bar is vLLM's 826 P90 on 4× B300 TP4 with DSpark. Target that.
