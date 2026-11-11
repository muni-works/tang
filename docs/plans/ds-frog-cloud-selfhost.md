# DeepSeek-V4.1-Flash on Frog Cloud: DSpark plan and self-host break-even

2026-11-11. The question: at what daily output-token volume does serving DS-V4.1-Flash
ourselves on 4× or 8× H200 cost less than DeepSeek's off-peak API?

## Prices assumed

| | price |
|---|---|
| DeepSeek direct, off-peak, output | $0.60 / M |
| input, cache miss | $0.15 / M |
| input, cache hit | $0.003 / M |
| H200 SXM on RunPod (secure) | $5.29 / GPU-hour (the rate our pods bill) |

- 4× H200 costs **$507.84 a day**.
- 8× H200 costs **$1,015.68 a day**.

## Break-even

Upstream cost per output token depends on how much input comes with it. There are two cases:

- **Output only:** $0.60 / M.
- **Agent mix:** an agent turn sends about 20 input tokens per output token, with about 85% of them cache hits. That costs `0.60 + 20 × (0.15 × 0.15 + 0.85 × 0.003)` = **$1.10 per M output**.

| node | workload | break-even output / day | sustained output tok/s |
|---|---|---|---|
| 4× H200 | output only | 846 M | 9,800 |
| 4× H200 | agent mix | 461 M | 5,340 |
| 8× H200 | output only | 1,693 M | 19,600 |
| 8× H200 | agent mix | 923 M | 10,680 |

"Sustained" is averaged over all 24 hours. Upstream is only this cheap off-peak, so a node that sits idle at night has to clear a higher bar by day.

## Can a node sustain that?

At large batch, the decode step is memory-bound on expert weights:
- Every rank reads its whole expert shard each step.
- The tang plan (`ds_shard_load::real_full_boxes_load_native`, all 40 layers, world 4) stores **74.2 GB of native weights per rank**.

Roofline ceilings, one step per batch:

| node | weight read per step (4.8 TB/s per GPU) | ceiling at batch 256 |
|---|---|---|
| 4× H200 | 74.2 GB / 4.8 TB/s ≈ **15.5 ms** | ≈ **16.5k tok/s** |
| 8× H200 | 37 GB / 4.8 TB/s ≈ **7.7 ms** | ≈ **33k tok/s** |

Why the other limits shouldn't bind at batch 256:
- **Compute:** about 26 GFLOP a token (≈13 B active) × 16.5k tok/s ≈ 430 TFLOP/s over 4 GPUs, about 107 TFLOP/s each. That's well under fp8 peak, but grouped expert GEMMs rarely get past 30–50% of peak.
- **Collectives:** 80 all-reduces of 256 × 5120 activations a step is about 0.2 GB of ring traffic, roughly 0.7 ms on NVLink.

Read against the break-even table:
- **4× H200** pays off only for the agent mix: 5.3k tok/s needs about 33% of roofline, around the clock.
- **4× H200 output-only** needs 60% of roofline, around the clock. That's unlikely.
- **8× H200** needs the same fraction of its own roofline as 4× (the ceiling doubles with the cost) and has more KV room for batch. It also clears the agent mix at about 33% utilization of roofline.

What has actually been measured:
- **Single stream:** the hand engine (`tang-llm`, 8× H100 EP) does **193 tok/s greedy** with DSpark.
- **Batched decode** on the derived step has never been run. Neither has a 40-layer derived run on a node.

**Recommendation:** keep DeepSeek upstream for now. Self-host once two things hold:
1. A batched derived step reaches ≥30% of roofline. That means ≥5k tok/s output on 4× H200, measured.
2. Frog Cloud's agent traffic sustains about 460 M output tokens/day.

Until then, a node at 20% load costs about 5× upstream per token.

## DSpark speculative plan

DSpark is the checkpoint's own drafter:
- 3 MTP stages that read target hiddens from layers 37–39.
- A 128-expert, top-3 MoE in each stage.
- A rank-256 Markov head.
- Blocks of 5 slots.

It already exists as a graph (`ir_ds_spark`, draft → verify through derived steps, checked against the f64 reference at real size).

**Measured** (hand engine, 8× H100, 200 tokens):
- **4.65 tokens per verify step** with 6 drafts (162/252 accepted).
- 27–28 ms a step.
- 193 tok/s greedy, versus about 120 tok/s without speculation.

**Plan for the derived path:**
1. **Verify step:** the main model's derived step at `k = 6` (`Rows::All`), sharded by the same EP/TP plan. A `k = 6` step reads the same weights as `k = 1`, so at small batch it costs about one step and yields about 4.6 tokens.
2. **Draft step:** the DSpark graph's derived step. It's small: 3 stages of a 128-expert top-3 MoE. Replicate it on every rank, with its experts expert-parallel like the main model's if it has to fit. It drafts from the accepted rows only, so it never rolls back.
3. **Batching:** speculation multiplies tokens a step only while the step is memory-bound. Near the batch-256 roofline it buys little. So the serving policy is speculation at low load (latency, and off-peak utilization) and plain batched decode at high load. Speculation does not change the break-even table above. It changes how fast a lightly loaded node answers.
4. **Gate:** greedy DSpark output must equal plain greedy output token for token, at world 4 on real weights, before it serves.

## What tang still needs before any of this is real

- **NCCL timing of the native sharded step on a single NVLink node.** The step exists (muni-works/tang#237) and passes at world 2/4 on one H200 with host collectives.
- **Device prefill**, so prompts don't go through the f64 interpreter.
- **A cached 40-layer plan.** Before #237's planner change, planning took 99 minutes on one core.
- **A batched derived step (`k > 1` rows of different requests)** and a roofline measurement of it.
