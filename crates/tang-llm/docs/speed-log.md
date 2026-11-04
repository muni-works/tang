# Speed log

Decode speed of `main` after each performance change, measured on rented GPUs. One row per
(change, GPU, model, streams). `bench_batch --q4` (greedy, no speculation, 256 tokens for 8B
and 128 for 32B, `mlx-community/Qwen3-*-4bit`) unless the row says zest-harness. tok/s is
total across streams. Speed of light is the weight bytes a step over HBM bandwidth (8B ~4.26 GB,
32B ~18.4 GB; H200 4.8 TB/s, H100 SXM 3.35 TB/s): 8B 1120 tok/s on an H200 and 790 on an
H100, 32B 260 and 182. % is of that, for one stream. Rows compare against the previous row
for the same GPU, model and streams; GPUs differ run to run (±5%), so treat changes under
~5% as noise unless the same pod measured both.

| date | commit | change | GPU | model | streams | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | bfcb90f | baseline | H200 | 8B | 1 | 121 | 10.8% | |
| 2026-10-20 | bfcb90f | baseline | H200 | 8B | 8 | 603 | | |
| 2026-10-20 | bfcb90f | baseline | H200 | 32B | 1 | 42.9 | 16.5% | |
| 2026-10-20 | bfcb90f | baseline | H200 | 32B | 8 | 172 | | |
| 2026-10-20 | c623ce1 | #80 GEMV fill, #81 pool, #82 attention splits, #83 rms_norm | H200 | 8B | 1 | 208 | 18.5% | +72% |
| 2026-10-20 | c623ce1 | (same) | H200 | 8B | 8 | 709 | | +18% |
| 2026-10-20 | c623ce1 | (same) | H200 | 32B | 1 | 69.5 | 26.7% | +62% |
| 2026-10-20 | c623ce1 | (same) | H200 | 32B | 8 | 215 | | +25% |
| 2026-10-20 | c623ce1 | (same, new pod) | H100 | 8B | 1 | 197 | 24.9% | |
| 2026-10-20 | c623ce1 | (same) | H100 | 8B | 8 | 700 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 8B | 16 | 846 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 1 | 67.6 | 37.1% | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 8 | 213 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 16 | 232 | | |
| 2026-10-21 | 701af6c | #87 speculation ≤2 streams, #89 tensor-core GEMM (no change at 1/8 rows in bench_batch, as expected) | H200 | 8B | 1 | 207 | 18.5% | −0.5% |
| 2026-10-21 | 701af6c | (same) | H200 | 8B | 8 | 711 | | +0.3% |
| 2026-10-21 | 701af6c | (same) | H200 | 32B | 1 | 70.7 | 27.2% | +1.7% |
| 2026-10-21 | 701af6c | (same) | H200 | 32B | 8 | 219 | | +1.9% |

AA-style single stream (`tang-llm serve`, speculation on as deployed, ~850-token prompt, 500
output tokens, temperature 0.7, streamed; output tok/s from first token to last, median of 3;
the probe is `docs/bench/speed_bench.py`):

| date | commit | GPU | model | AA output tok/s | bar (fastest public API) |
|---|---|---|---|---|---|
| 2026-10-21 | 701af6c | H200 | 8B | 200.5 | Eigen 303 |

Measured on the same H100 pod, waiting to merge (each on top of c623ce1):

| change | model | streams | tok/s | vs c623ce1 |
|---|---|---|---|---|
| draft only at ≤2 streams (zest-harness, speculation on) | 8B | 4 / 8 / 16 | 408 / 542 / 600 | +38% / +61% / +91% |
| (same) | 32B | 8 / 16 | 149 / 160 (≤4) | +58% / +86% |
| device top-64 sampling | 8B | 8 / 16 | 785 / 1002 | +12% / +18% |
| (same) | 32B | 8 / 16 | 221 / 243 | +4% / +5% |
| tensor-core 4-bit GEMM, 9-32 rows | 8B | 16 | 895 | +6% |
| (same) | 32B | 16 | 282 | +21% |
| all three | 8B / 32B | 16 | 1051 / 298 | +24% / +28% |

## gpt-oss-20b

Single stream, AA-style (a ~1k-token chat prompt, output tok/s; `scripts/gpt_oss_speed.sh`).
The bar is Groq's claimed ~1,200 tok/s.

**Bytes per decoded token** (what a step must read; gpt-oss-20b, 24 layers):

| part | format | per token |
|---|---|---:|
| experts: 4 of 32 per layer, gate/up 5760×2880 + down 2880×2880 | MXFP4 (4.25 bits) | 1.27 GB |
| attention Q/K/V/O, 26.5M a layer | bf16 | 1.27 GB |
| output head 201088×2880 | bf16 | 1.16 GB |
| KV at ~1.3k context (12 full layers ~1.3k keys, 12 sliding 128), f32 | | ~0.05 GB |
| router, norms, biases | f32 | ~0.01 GB |
| **total** | | **~3.75 GB** |

Speed of light at that size: 1280 tok/s on an H200 (4.8 TB/s), 890 on an H100, 270 on a 4090.
Groq's 1,200 is ~94% of the H200's. **To pass it, the weights have to shrink:** attention and
the head at 4 bits (Q4, as the Qwen rows run) would take a token to ~1.95 GB, a 2460 tok/s ceiling
on the H200. The experts are already 4-bit.

**Per kernel** (4090 nsys, 256 decoded tokens, before the 4-rows-a-warp GEMV): the expert GEMV
was 48% of GPU time at ~350 GB/s (a third of peak) and the bf16 GEMVs 36% at ~0.94 TB/s (at
peak). Sink attention was 6%, rms_norm 4%. Host side: ~460 launches a token, with a
host-to-device upload (ids, block tables) and a logits download every step.

| kernel | bytes/token | at 4.8 TB/s | measured | what's left |
|---|---:|---:|---|---|
| `moe_gemv_mxfp4` (48 calls) | 1.27 GB | 0.26 ms | 4090: 3.6 ms → ~2.2 ms after 4 rows a warp | input staged in smem (new, A/B below); more columns a block |
| `gemv_bf16` attention (48 calls) | 1.27 GB | 0.26 ms | at bandwidth | 4-bit attention weights |
| `gemv_bf16` head (1 call) | 1.16 GB | 0.24 ms | at bandwidth | 4-bit head, or device top-k so only the argmax comes back |
| `attn_sink_partial` + combine | ~0.05 GB | 0.01 ms | 6% (launch-bound) | fold into fewer launches |
| everything else (~360 launches) | | | ~1.5 ms on H100 | CUDA graphs |

Measured so far (before the rebase onto #88/#89):

| date | commit | change | GPU | model | streams | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | 29221a7 | #95, 13-token prompt | RTX 4090 | gpt-oss-20b | 1 | 157 | 58% | |
| 2026-10-20 | 1b8474c | #95, 13-token prompt | H100 | gpt-oss-20b | 1 | 182 | 20% | |
| 2026-10-20 | 1b8474c | #95, 13-token prompt | H100 | gpt-oss-120b | 1 | 119 | | |

**Plan for when gpt-oss is back in focus** (on hold since 2026-10-21; 30% of zest goes to
speed, all of it on Coder-Next and tang-pro for now). In order, each with what's ready:

1. **Q4 attention and head**, the path past Groq: 3.75 → ~1.95 GB a token, H200 ceiling
   1280 → ~2460 tok/s. `--q4` already quantizes attention, the embedding and the head at load
   (affine 4-bit, group 64) and the 4-bit GEMV serves them. Correctness checked offline:
   `scripts/gpt_oss_q4_check.py` applies the same rounding to transformers' weights, and tang
   `--q4` matches it at max KL 1.1e-4 (tiny model, CPU). Still unmeasured: what Q4 costs the
   real 20b. `gpt_oss_speed.sh` measures it (`--q4` vs bf16 KL, 256 positions) beside the speed.
   If the head costs too much, keep the head bf16 and quantize attention only (1.27 GB saved of
   the 1.8), or try group 32.
2. **The MXFP4 GEMV A/B** (input via L1, the default, vs staged in shared memory,
   `TANG_MOE_GEMV=smem`), in the same script. The smem kernel has not run on a GPU yet.
3. **CUDA graphs over the decode step**: the H100 step is launch-bound (~460 launches a token).
4. **Speculation**: already exact for gpt-oss (`tests/speculate.rs` passes on CUDA).

One H200 hour covers 1 and 2 (`scripts/gpt_oss_speed.sh <gpt-oss-20b dir>`; it prints these
rows and an nsys summary).

## Dead ends

Tried and measured at under 3% (or worse):

- **4-bit GEMV, a warp a column, x read through L1 (no shared-memory stage):** 30% slower at one
  row, worse with more rows (L1 traffic is 8x the weight bytes per extra row).
- **GEMV with the next stage's weights double-buffered in registers:** 5-15% slower; register
  pressure cut occupancy.
- **GEMV, 2 columns a warp instead of 4:** 2-10% slower.
- **GEMV K-split sizing, 1 / 4 / 8 / 16 blocks an SM:** within 3% of each other once the
  128-bit loads are in; 4 kept.
- **GEMV with the whole K slice of x resident in shared memory and every weight load issued up
  front (`gemv_q4_xr`, 2026-10-21, H200):** 8B solo 207 → 130 tok/s, 32B 71 → 36. Registers
  (16 uint4 of weights a lane) leave one 8-warp block an SM, and a block loads everything,
  then computes: no overlap between one block's math and the next one's loads. The fix has to
  keep several blocks (or a load pipeline) per SM.
- **Device top-k for a lone stream:** 1.5% slower than sampling its one row on the host;
  only batches cut on the device.
- **The VRAM-pressure check on pool misses only (#81)** measured +1.5% alone. It's kept because
  it takes ~2.6 ms of CPU off each step, which matters once kernels stop hiding host time.

## Qwen3-Coder-Next (qwen3next Flash engine)

`Qwen3-Coder-Next-UD-Q4_K_XL.gguf` (unsloth, 49.6 GB), the Flash engine's qwen3next path
(`src/flash/coder.rs`), one stream, greedy. A step reads 3.42 GB at a 1.3k context (3.52 GB at
4k): dense weights, 10 routed + 1 shared expert a layer, head, K/V. Speed of light is that over
HBM bandwidth: 1,400 tok/s on an H200, 970 on an H100 SXM.

AA rows are `flash-aa`, Artificial Analysis-style: a ~1.24k-token chat prompt (one of
`explain` / `refactor` / `write`, the same 140-line Rust file with a different task), 512
output tokens, tok/s after the first token. The fastest public API for this model is Novita at
174 tok/s.

| date | commit | change | GPU | prompt | drafts | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | c33c4df | engine (#84), flash-bench 4k ctx | H100 | code ×4k | none | 156 | 16.1% | |
| 2026-10-20 | b9fbcad | fusions, K-split GEMV, 1024-thread norms (#90) | H100 | code ×4k | none | 198 | 20.4% | +27% |
| 2026-10-21 | d9f1d29 | (same kernels) AA | H200 | explain | none | 205.5 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | refactor | none | 205.3 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | write | none | 205.3 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | explain | suffix | 205.1 | | 1.15 tok/window |
| 2026-10-21 | d9f1d29 | | H200 | refactor | suffix | 340.5 | | 2.71 tok/window |
| 2026-10-21 | d9f1d29 | | H200 | write | suffix | 218.7 | | 1.28 tok/window |
| 2026-10-21 | d9f1d29 | separate SiLU launch (now the default) | H200 | explain | none | 217.4 | 15.5% | +2% (noise) |
| 2026-10-21 | 4ac3171 | K-split GEMV, 2 warps a row (was 4) | H200 | explain | none | 227.2 | 16.2% | +4.4% |
| 2026-10-21 | 4ac3171 | (same pod, 2 warps for 4k+-row matrices only) | H200 | explain | none | 226.9 | 16.2% | +4.3% |
| 2026-10-21 | 4ac3171 | (same build, AA) | H200 | refactor | suffix | 370.2 | | 2.71 tok/window |
| 2026-10-21 | 4ac3171 | | H200 | write | suffix | 229.0 | | 1.25 tok/window |
| 2026-10-21 | 4ac3171 | | H200 | explain | suffix | 222.3 | | 1.16 tok/window |
| 2026-10-21 | 2a723ae | (round-3 pod, defaults of 4ac3171) | H200 | explain | none | 222.3 | 15.9% | pod baseline |
| 2026-10-21 | 2a723ae | fast Q4_K/Q5_K expert dot (32 weights a lane) | H200 | explain | none | 230.9 | 16.5% | +3.9% |
| 2026-10-21 | 2a723ae | Q8X dense repack (aligned codes, separate scales) | H200 | explain | none | 231.8 | 16.6% | +4.3% |
| 2026-10-21 | 8fdfd13 | Q8X + fast expert dot (the defaults now) | H200 | explain | none | 240.8 | 17.2% | +8.3% |
| 2026-10-21 | 2a723ae | + bf16 router (opt-in) | H200 | explain | none | 243.1 | 17.4% | +1% |
| 2026-10-21 | 2a723ae | (Q8X + bf16 router, AA) | H200 | refactor | suffix | 403.5 | | 2.71 tok/window |
| 2026-10-21 | 2a723ae | | H200 | write | suffix | 252.0 | | 1.25 tok/window |
| 2026-10-21 | eed9069 | (round-4 pod, defaults) | H200 | explain | none | 242.0 | 17.3% | same as round 3 |
| 2026-10-21 | 625db0d | (round-5 pod, defaults: Q8X loads two steps ahead) | H200 | explain | none | 241.5 | 17.2% | 0% |
| 2026-10-21 | c4aad43 | two expert-down rows a warp (fastk=2, now the default) | H200 | explain | none | 246.7 | 17.6% | +2.2% |
| 2026-10-21 | 625db0d | (same pod, AA) | H200 | write | suffix | 247.2 | | 1.25 tok/window |

Stage sweep on that H200 pod (`TANG_CODER_STAGES=1`: each stage alone in a CUDA graph, replayed;
µs a step at a 1.3k context, window 4,856 µs): routed+shared gate|up 876 (771 GB/s), down 801
(501 GB/s), GDN input GEMV 618 (1.6 TB/s), GDN output 301, top-k 286, GDN conv+step 238,
moe combine+norm 214, attention split-K 207 (154 GB/s), router 176, head 174, post-mixer norm
166, attention in/out 260. The stages sum to 4,461 µs, so launch gaps are ~8% of the window.

### Batch decode past 8 rows (2026-11-03, b957a2a9+, H200, explain prompt, ~1.2k context)

`TANG_BATCH_N=64 flash-batch-check`: bitwise PASS at 64 sequences (tokens and logits equal to
each sequence alone). Graphs, 16 steps:

| B | ms a step | tok/s aggregate | tok/s a stream |
|---|---|---|---|
| 1 | 4.03 | 248 | 248 |
| 2 | 4.73 | 423 | 211 |
| 4 | 6.49 | 616 | 154 |
| 8 | 9.86 | 812 | 101 |
| 16 | 21.99 | 728 | 45 |
| 32 | 40.53 | 790 | 25 |
| 64 | 73.90 | 866 | 14 |

Past 8 rows the step grows linearly with B. Eager profile at B=64 (ms a step): expert gate|up
23.6 (~0.46 GB a layer of distinct experts in 0.49 ms: ~0.94 TB/s), expert down 24.3 (~0.46
TB/s), GDN in 6.8, GDN conv+step 4.3, GDN out 4.1, attention split-K 2.6, head 1.9, attention
in 1.7, router 1.6, attention out 1.3. The grouped expert kernels (`cx_moe_gu_g` /
`cx_moe_down_g`, ~1.75 tokens an expert) and the 32-row K-split GEMVs (`cx_gemvkb`, L2-bound
re-reading activations per weight row) are the walls.

- **Tensor-core GEMMs for batch windows** (`TANG_CODER_BGEMM=1`: `cx_gemm_dense` +
  `cx_moe_gemm_*`, the prefill kernels): slower at every B past 8 (B=64: 132.5 vs 78.3 ms
  eager) and not bitwise. They're tiled for 64 tokens an expert; decode has ~2.

### Streaming grouped experts for batch windows (2026-11-03, H200, explain, ~1.2k context)

`cx_moe_gu_s` / `cx_moe_down_s`: grid (row tiles of 64 / 128 rows, groups of at most 8 pairs),
each warp walking its rows of one expert with cp.async staging, the group's activations staged
once a block, and `cx_dotq_m` decoding each Q4_K/Q5_K weight block once for all the group's rows
(each row's arithmetic the single-row dot's). Bitwise PASS at 64. Graphs, 16 steps:

| B | ms a step | tok/s aggregate | tok/s a stream |
|---|---|---|---|
| 1 | 4.03 | 248 | 248 |
| 2 | 4.74 | 422 | 211 |
| 4 | 6.51 | 615 | 154 |
| 8 | 9.86 | 812 | 101 |
| 16 | 20.01 | 800 | 50 |
| 32 | 31.04 | 1031 | 32 |
| 64 | 51.16 | 1251 | 20 |

Eager, B=64 (ms a layer, GB/s from ~367 distinct experts + shared): gate|up 0.57 → 0.29
(~1.5 TB/s; 0.22 ms / ~2 TB/s on 64-token prompts), down 0.50 → 0.23 (~0.95 TB/s; 0.18 /
~1.2 TB/s). Ring depth barely matters (gate|up best at 1 stage — occupancy wins; down 2).

- **Plain code sums precomputed per 16 bytes** (dropping 8 of 16 dp4a a row-token, exact): slower
  (gate|up 0.22 → 0.26 ms), the extra prologue pass costs more than it saves.
- **Deeper rings / more rows a warp for down** (4/8 stages, 32 rows): flat or worse.

### Tensor-core batch windows, tolerance-gated (2026-11-03, H200, explain, ~1.2k context)

Past 8 rows, batch windows may drop bitwise equality: `flash-batch-check` then gates on
**greedy agreement given the same history ≥ 99%** (each sequence's steps up to and including
its first divergence from the sequence alone) and **mean |Δlogit| ≤ 0.2** over those steps
(`TANG_BATCH_TOL`); max |Δlogit| is reported. Narrow windows (T ≤ 8) stay bitwise.

Measured noise floor at B=64 (16 steps): agreement 99.50% (986/991), mean |Δlogit| 0.122,
max 11.9. The int8 MMA down kernel matches the per-pair dot to ~1e-6 relative
(`TANG_CODER_MMA_CHECK=1`), so the spread isn't the kernel: a reordered fp32 sum flips the int8
rounding of some activation downstream, and the flips compound over 48 layers. A 0.05 max bar
is unreachable for any change of summation order in this pipeline.

| commit | change | B=16 | B=32 | B=64 ms | B=64 tok/s |
|---|---|---|---|---|---|
| 34c91db3 | streaming experts (bitwise) | 20.01 | 31.04 | 51.16 | 1251 |
| 90a132d9 | + MMA down experts (`cx_moe_down_m`) | 19.32 | 29.75 | 48.64 | 1316 |
| 90a132d9 | + Q8X dense MMA (`cx_mma_q8x`) | 18.60 | 27.24 | 43.02 | 1488 |
| 94c1b95b | + mixed matrices (GDN in on MMA) | 18.48 | 25.79 | 40.01 | 1600 |

Eager B=64 at 94c1b95b, ms a step: gate|up (streaming) 10.7, down (MMA) 5.8 (~1.8 TB/s), GDN in
4.1, GDN conv+step 4.3 (≈ the f32 state's bandwidth at 64 sequences), router 1.6, route+plan
1.6, GDN out 1.35, attention in/out + head ~2.

- **Gate|up on MMA** (`TANG_CODER_MOE_MMA=all`): no faster than streaming (10.2–11 vs 10.7 ms).
- **Narrow streaming experts** (`TANG_CODER_NSTREAM=1`, T = 2..8, bitwise, spec-test PASS):
  slower. T=8 oracle window 10.0 → 13.5 ms, forced-wrong windows 7.2 → 9.3 ms; T=1 unchanged
  (4.6 ms). One sequence gives ~1 token an expert, so grouping saves no bytes, and the
  streaming kernels stage activations per block with fewer warps than per-pair `_d4`.

### Batch windows: plan prefix, router, expert MMA retries (2026-11-04, H200, new pod)

| commit | change | B=16 | B=32 | B=64 ms | B=64 tok/s |
|---|---|---|---|---|---|
| 5bbaee5c | parallel prefix in `cx_moe_plan_w` (route+plan 1.6 → 0.8 ms) | 17.41 | 24.74 | 38.78 | 1650 |

Tolerance PASS (99.50%, mean |Δlogit| 0.122). Eager B=64, ms a step: gate|up 10.6, down 5.6,
GDN conv+step 4.35, GDN in 4.06, router 1.64, GDN out 1.35, route+plan 0.79, head 0.76.

- **Expert MMA with precomputed code sums + a superblock's scales unpacked once** (68896d53):
  slower, down 5.8 → 7.5 ms, gate|up unchanged. Reverted.
- **Two 16-row tiles a warp in the expert MMA** (B fragment and scales shared): gate|up 10.9 →
  12.5 ms. Dropped.
- **Router with the activations in shared memory** (`cx_router_b`, `TANG_CODER_ROUTERB=1`):
  1.64 → 1.89 ms. Off.
- Gate|up stays ~0.22 ms a layer on all three kernels (per-pair-dot streaming, MMA, MMA with
  two tiles): ~2 TB/s for ~400 distinct experts × 1.18 MB. Down is ~1.9 TB/s on MMA.

### Dead ends (Coder-Next)

- **22 → 12 launches a layer with no faster kernels:** +2% (156 → 159 on H100). The step is
  kernel-latency-bound, not launch-bound.
- **Expert GEMVs, 4 rows a warp** (all staged before decoding): −4% vs 2.
- **K-split GEMV with 8 warps a row** instead of 4: −7%. Both together: −10%.
- **SiLU + int8 of h fused into the down kernel:** −2% vs a separate launch (each down block
  recomputes its pair's 512 values; the extra latency outweighs the saved launch).
- **The eager event profiler** (`TANG_CODER_PROF`) put about 2× the graph's time on small
  kernels. Use the stage sweep for decisions.
- **Top-k in the router GEMV's last block** (atomic counter, one launch fewer): −2% (217 → 213).
  The last block's serial top-k costs more than the launch it saves.
- **K-split routed experts** (Q warps a row, unstaged, like `cx_gemvk`): gate|up −4% / −6% /
  −16% at Q = 1 / 2 / 4, down +0.5% / −6% / −18%. Experts want the staged rows; the dense
  matrices want the split.
- **K-split GEMV, 1 warp a row** (no split): −1.5% vs 4, −6% vs 2.
- **A Qwen3 drafter** (estimate, not built): teacher-forced top-1 agreement with the engine's
  greedy output is 58 / 88 / 71% for Qwen3-0.6B and 62 / 90 / 77% for 1.7B
  (explain / refactor / write). A verify window of 8 costs ~3× a T=1 step here (each token
  routes to its own experts), so a drafter only pays on refactor-like text, which the free
  suffix drafter already covers (2.7 tokens a window). Expected gain on fresh text ≤10%, for a
  second model engine. Not worth building yet.
- **First rows of a `flash-aa` run** read ~4% low (205 vs 213–217 for the same kernels later in
  the run). Compare variants inside the A/B block, which warms up first.
- **bf16 router weights** (`TANG_CODER_ROUTER16=1`): +1% (240.8 → 243.1), under the bar. The
  router is 4.2 MB a layer of F32 but reads at ~1.1 TB/s either way. Kept opt-in.
- **Top-k in the expert gate|up kernel's prologue** (every block of a token reruns it; no
  `route` launch): −9% (242 → 220, eed9069). The top-k latency now sits in front of every
  expert block.
- **Norm kernels at 512 / 256 threads** instead of 1024: −4% / −11%.
- **Attention split-K chunks of 32 / 16 positions** (from 64): 0% / −3%. At 16 the merge
  costs more (218 µs) than the split saves (153 vs 224 µs).
- **The shared expert on a parallel graph branch** (forked before the router, joined before
  the combine, so it overlaps the router and the one-warp top-k): −2% (241.5 → 236.9). The
  fork/join costs more than the overlap gives.
- **Q8X GEMV with both 512-wide steps' loads issued first:** 0% (241.5 vs 242). Left in (harmless).
- **Running `flash-ref` while benchmarking:** flash-ref uses the GPU too, and halved every
  kernel's speed for the length of the overlap. Run references before or after, never during.
- **nsys / ncu on the RunPod image:** ncu's injection lacks `cuTensorMapEncodeIm2colWide`,
  which cudarc resolves at load, and nsys isn't installed (apt fails).

### Wide prefill: k32 expert/dense GEMM tile, warp-per-column GDN, f32 router GEMM (2026-11-04, H200)

`flash-wide-check --ctx 3000` (refactor ids, `TANG_CODER_WIDE=2048`, the serve width), GPU time
of the two wide prefills by stage (`TANG_CODER_PROF=1`; the narrow prefill is now excluded):

| stage | before ms | after ms |
|---|---|---|
| moe_gu (`cx_moe_gemm_gu`) | 1107 | 568 |
| GDN recurrence | 441 (`fl_gdn_conv_step_seg`, 32 blocks) | 190 (`fl_gdn_wide` + norm) |
| moe_down | 427 | 227 |
| gdn_in (`cx_gemm_dense` + GEMV fallback) | 408 | 243 |
| router (F32: 256 eight-row GEMVs a layer) | 393 | 89 (`cx_sgemm_wide`) |
| gdn_out / attn_in / attn_out | 137 / 92 / 46 | 91 / 59 / 30 |
| wide prefill, 3000 tokens | 1.63 s, **1,844 tok/s** | 0.80 s, **3,741 tok/s** |

1000 / 8000 tokens: 3,428 / 3,980 tok/s. Narrow prefill (`prefill_slot`, what `flash-aa` and
`--batch`-less paths use) is unchanged at ~810 tok/s.

- `gemm_tile32` (every k32-scaled type: Q4_K/Q5_K/Q8_0/Q8X/Q4_0/Q5_0/IQ4): one m16n8k32 MMA a
  32 columns, scales and mins once a 32 instead of per 16, the tokens' scales and per-32 code
  sums staged in shared memory (no shuffles in the loop), and the 32 columns permuted so each
  8-weight decode is used whole (half the decodes). Q3_K/Q6_K keep the k16 tile.
- `fl_gdn_wide`: 8 lanes per state column (columns are independent given k, q, g, b), 3-step
  shuffle reductions, no block barrier a token; gated norm in `fl_gdn_wide_norm`.
  `TANG_CODER_GDNW=0` for the old kernel.
- Router: F32 GEMM (`TANG_CODER_SGEMM=0` off); other non-GEMM matrices take the batched K-split
  GEMV (32 rows a launch) instead of 8-row GEMVs.

Serve, `/workspace/load.py 64 <port> 3` after `serve --batch 64 --ctx 32768`, same H200, HEAD
4133d39a vs this: aggregate **302 → 421 tok/s**, TTFT p50 25.6 → 16.5 s, p95 108 → 69 s, wall
406 → 292 s, 192/192 ok. Gates: `flash-wide-check` 3000/8000 PASS (KL 9e-7 / 3e-6); 1000 fails
the 1e-3 bar on both (base 1.15e-2, this 7.9e-3: pre-existing, FA2 at short context);
`flash-tcheck` t1 vs t8 bitwise; `flash-spec-test` PASS.

### Single-stream prefill takes the wide windows (2026-11-04, H200)

`Coder::prefill` (flash-aa, single-request serve) used only the narrow path (~850 tok/s) even
when wide windows existed. It now takes `prefill_wide` on slot 0 for prompts of at least
`TANG_CODER_WIDE_MIN` (128) tokens when wide windows are built, no MTP head is loaded and no
per-window logits are wanted. `flash-aa` builds 2048-row windows (`TANG_CODER_AUTO_WIDE`, 0 off;
+~50 s load); `serve` without `--batch` builds them too (`TANG_SERVE_WIDE`). Check tools keep
narrow references (auto-wide off by default in `load_coder`).

flash-aa explain (1235 tokens): prefill **1.49 → 0.35 s** (830 → 3,530 tok/s); decode unchanged.
Crossover (flash-wide-check): 32 tok 785 vs 472, 64: 856 vs 671, 128: 851 vs 1,141, 256: 849 vs
1,770, 512: 842 vs 2,589 tok/s (narrow vs wide).
