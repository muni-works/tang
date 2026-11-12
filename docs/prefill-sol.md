# Prefill speed of light: coder-next and DeepSeek-V4.1-Flash

2026-11-12. This doc asks two questions: what prefill costs per dollar on the GPUs we rent, and how fast DeepSeek-V4.1-Flash (DS) prefill could run on H200s.

Every number is tagged **[M]** if it was measured on our pods today or **[A]** if it is assumed or derived from arithmetic. Prices are RunPod secure cloud: RTX 4090 $0.89/hr and H200 $5.29/hr.

Why it matters: an agent sends about 25 prompt tokens per output token, and about 89% of them hit the cache. That leaves about 2.75 uncached prefill tokens per output token, so prefill speed sets Frog Cloud's margin.

## 1. GEMM throughput at prefill shapes [M]

`scripts/prefill_gemm_bench.cu` uses cuBLASLt. For each GEMM it takes the best of the heuristic's top 8 algorithms. Operands are pseudo-random (constant data draws less power and clocks higher). The layer shapes come from the configs:

- **coder-next:** 48 layers (36 GDN + 12 gated attention). Every layer is MoE with 512 experts, top-10, inter 512, plus a shared expert.
- **DS:** 40 layers, dim 5120, low-rank q/o, 384 experts top-6, inter 2304, plus a shared expert.

Routed experts run as a strided batch of E GEMMs with `M·topk/E` rows each, assuming perfect balance. The figures count dense and expert GEMMs only, about 6.5 GFLOP/token for coder-next and 30.1 for DS. They leave out attention cores, the GDN scan, the DS indexer and the LM head (prefill only needs the last row).

H200 A (`w2myvdc7bi8mke`) ran through the warm runner, job 1112-143119-a34c12. The 4090 (`zdwi8c9xzf4g5g`) ran directly.

GEMM-only tok/s for the whole model, best of int8 and fp8, with TFLOP/s achieved:

| model | M (tokens/chunk) | H200 tok/s | H200 TFLOP/s | 4090 tok/s | 4090 TFLOP/s |
|---|---|---|---|---|---|
| coder-next | 2,048 | 77,000 | 547 | 17,260 | 123 |
| coder-next | 8,192 | 166,500 | 1,083 | 39,020 | 254 |
| coder-next | 32,768 | 173,100 | 1,126 | 55,170 | 359 |
| DS | 2,048 | 13,450 | 405 | 2,970 | 89 |
| DS | 8,192 | 35,600 | 1,072 | 8,730 | 263 |
| DS | 32,768 | 47,660 | 1,435 | 16,090 | 484 |

bf16 runs at about 0.55–0.7× these rates on the H200 and about 0.5× on the 4090. Full per-shape rows are in the job logs.

**Expert starvation, measured.**

- At M = 2k, DS experts get m = 32 rows each and run at 244 TFLOP/s on the H200 and 55 on the 4090. At M = 32k they get m = 512 and run at 1,545 and 497.
- coder-next experts at m = 48 run at 360 / 291 TFLOP/s (gate_up / down) on the H200. At m = 640 they run at 1,052 / 742.
- Below roughly 8k tokens per chunk, the MoE weights are streamed for almost no work. tang's 256-token chunk is 30–60× below the knee.

## 2. $ per million prefill tokens

$/M tokens = $/hr ÷ (tok/s × 3600) × 10⁶. GEMM-only rates are a ceiling, because attention, the scan and the indexer add about 10–20% [A].

| model | setup | tok/s | $/M prefill | status |
|---|---|---|---|---|
| coder-next | tang today, H200, 4k ctx (chunk 256) | 1,856 | $0.79 | [M] #234 |
| coder-next | tang grouped experts, chunk 1024 (#263) | 3,174 | $0.46 | [M] #263 |
| coder-next | llama.cpp, same H200 | 5,688 | $0.26 | [M] #234 |
| coder-next | H200 GEMM SOL, chunk 8k | 166,500 | $0.0088 | [M] GEMMs, [A] rest |
| coder-next | H200 GEMM SOL, chunk 32k | 173,100 | $0.0085 | [M] GEMMs, [A] rest |
| coder-next | 1× 4090, weights resident (doesn't fit) | 55,170 | $0.0045 | [M] GEMMs |
| coder-next | 3× 4090 layer pipeline (46 GB over 72 GB) | 55,170 | $0.0045 | [M] per-card GEMMs, [A] pipeline at 1× |
| coder-next | 1× 4090, PCIe layer streaming, 32k/pass | 4,770 | $0.052 | [M] 6.7 GB/s H2D, [A] overlap |
| coder-next | 1× 4090, streaming, 256k/pass | 37,300 | $0.0066 | [A] from the same measurements |
| DS | tang today, 2-layer native step ×20 → 40 layers, 1 H200 | see §4 | — | [M] 2 layers, [A] ×20 |
| DS | H200 GEMM SOL, chunk 32k, per GPU | 47,660 | $0.031 | [M] GEMMs, [A] rest |
| DS | 4× H200 node (EP4) | ~160,000 | ~$0.037 | [A] (§3) |
| DS | 8× H200 node (EP8) | ~320,000 | ~$0.037 | [A] (§3) |
| DS | 4090 | — | — | 510 GB native needs ~25 cards; streaming 307 GB at 6.7 GB/s = 46 s/pass. No. |

**Layer streaming on the 4090.** The pod's link trains at PCIe Gen4 **x4**, and pinned host-to-device copies measured 6.7 GB/s [M]. Streaming coder-next's 46 GB once per pass takes 6.9 s. Compute at 32k tokens takes 0.59 s, so a pass is stream-bound until it holds about 380k tokens. An x16 Gen4 host (~25 GB/s [A]) would break even near 100k tokens per pass. Layer-wise prefill in tang (#263) is grouping, not weight streaming. On the H200, grouped experts take coder-next from 1,876 to 3,174 tok/s at chunk 1024. PCIe streaming stays unmeasured beyond the copy rate.

**Multi-card.** Pipelining layers over three 4090s keeps each card at its single-card rate, and the activations crossing between cards are tiny (2048 × 2 B per token). So $/token equals the one-card GEMM rate: $0.0045/M. That is 1.9× cheaper per GEMM FLOP than an H200 at chunk 32k ($0.0085/M) [M+A]. Coder-next's 46 GB is Q4, so the cards would dequantize to int8 tiles in flight.

**Verdict: disaggregated prefill on cheap GPUs is not worth building now.**

- At the speed of light, 4090s beat an H200 on $/token by at most about 2×.
- Our real gap is software. tang today costs $0.79/M against an H200 GEMM ceiling of $0.0085/M, which is ~90×. llama.cpp on the same box is ~30× above the ceiling.
- On the cheap cards, PCIe (x4 here) and the 24 GB of memory push us into multi-card pipelines. That doubles the operational surface for a 2× win.
- Revisit once tang reaches about 30% of the H200 GEMM SOL (~50k tok/s on coder-next).

What gets us there on the H200:

1. Chunks of 8k–32k tokens instead of 256, which moves experts from m ≈ 5 to m ≈ 160–640.
2. Grouped expert GEMMs at those m.
3. The remaining ~10–20% from fused attention, the chunked GDN scan and the indexer.

## 3. DeepSeek-V4.1-Flash prefill plan (per GPU, per 4×/8× H200 node)

### FLOPs per token [A, from config.json]

| part | GFLOP/token |
|---|---|
| dense + shared + routed-expert GEMMs (counted in §1) | 30.1 |
| sparse attention core: 64 heads × 512 dim over index_topk 512 + window 128, ×40 layers | ~3.4, flat in context |
| indexer: 32 heads × 128 dim against compressed keys, 8 indexer layers | ~0.5 at 32k, ~2.1 at 128k (assumes 4× key compression) |
| **total** | **~34 at 32k, ~36 at 128k** |

The indexer is a dense q·kᵀ over every compressed key, so it is the only term that grows with context. At 128k it adds ~6% to the work. It runs in fp8, at the same rate as a GEMM with M = chunk and N = keys/4. Selection (top-512 of up to 2,048 candidate blocks) is bandwidth-cheap. The CSA/HCA compressors are bf16 and small.

### Expert starvation

Rows per expert = T·6/384 = T/64, where T is the global tokens in a chunk.

- An fp8 weight tile is compute-bound when m ≳ peak FLOPs ÷ (2 × bandwidth) ≈ 1,979 T ÷ (2 × 4.8 TB/s) ≈ 206 rows [A]. An fp4 tile needs ≈ 103 rows.
- So T ≥ ~6.6k tokens per expert step with fp4 experts, or ~13k with experts dequantized to fp8.
- The measured knee in §1 matches: 1,072 TFLOP/s at T = 8k and 1,435 at 32k, against 405 at 2k.
- Expert parallelism does not change rows per expert. Only the global chunk does. EP8 with a 32k chunk gives every GPU 48 experts × 512 rows.

### All-to-all for EP during prefill [A]

- Each token in each MoE layer dispatches 5,120 × 1 B (fp8) to 6 experts and combines 5,120 × 2 B (bf16) back: about 92 KB per token per layer, or 3.7 MB per token over 40 layers. Node-local dedupe lowers this.
- **Within one node:** at 40k tok/s per GPU, with 7/8 of the traffic remote, each GPU moves about 130 GB/s against ~450 GB/s per direction on NVLink. That is not binding, and it overlaps with expert compute (two micro-batches).
- **Across nodes** (400 Gb/s IB ≈ 50 GB/s per GPU): the ceiling is ~14k tok/s per GPU, so it is binding. DS prefill should stay inside one node.

### Memory fit [M, #227/#237]

- Native weights total 510 GB: experts 296 GB, Engram tables 203 GB, the rest ~11 GB.
- With Engram rows in host memory (as #227 planned), experts plus the rest come to 307 GB. That is ~77 GB per GPU on 4× H200 and ~38 GB per GPU on 8×.
- 8× leaves ~100 GB per GPU for KV and the 32k-token activation chunk.
- A single H200 cannot hold the experts, so a one-GPU deployment would stream fp4 experts over PCIe. At ~50 GB/s Gen5 [A] that is ~6 s per pass, and it breaks even with compute only at ~280k tokens per pass. Not a serving configuration.

### Speed-of-light targets

| setup | tok/s | $/hr | $/M prefill |
|---|---|---|---|
| 1× H200 (compute only, if it fit) | ~42,000 (47.7k GEMM ÷ ~1.13 for attention + indexer) | 5.29 | ~$0.035 |
| 4× H200, EP4 + DP attention, chunk 32k | ~160,000 | 21.16 | ~$0.037 |
| 8× H200, EP8 + DP attention, chunk 32k | ~320,000 | 42.32 | ~$0.037 |

DSpark (MTP draft) is decode-only and adds nothing to prefill. Engram lookups are host-memory gathers of a few KB per token per Engram layer. They are hidden behind the GEMMs if they are prefetched one layer ahead, which needs pinned host memory and about 1 GB/s.

### What tang has on main

| PR | what it gives | prefill gap |
|---|---|---|
| #220 / #222 | real-size DS graph, derived step, CPU reference (3.9e-15) | prefill runs on the f64 interpreter; there is no device prefill |
| #227 | native fp8 (`Fp8B32`) dense + MXFP4 routed experts, int8 GEMV with Q8x32 activations; H200 2-layer step 2.42 ms vs 5.40 f32 | GEMV-shaped, so k tokens become k GEMV rows with no MMA tiling |
| #237 | per-rank shard loader: full 384 experts + full vocab, N=4 at 3.71 GB per rank for layers 0,2, bit-exact | rank programs are f32 only (no native formats yet); world 4 needs a non-star network (see ds-4gpu notes) |
| #223 (open) | multi-GPU TP/EP plan; real plan 545 tok/s on 4× H200, 677 on 8× | indexer weights replicated; NCCL path never run on GPU |

The plan, in order:

1. A device prefill path: the derived k-token step with `k` = chunk, as §4 exercises.
2. Grouped fp4/fp8 expert MMA at m ≥ 128 instead of the int8 GEMV.
3. A fused sparse-MLA flash kernel over the indexer's top-k.
4. Native formats in the #237 rank programs, then EP all-to-all on NVLink in one node.

## 4. Measured: DS 2-layer chunked prefill on H200 A [M]

`examples/ds_real_gpu.rs` gains three knobs:

- `DS_STEP_K=k`: the derived step appends k tokens, so tok/s = k ÷ step time.
- `DS_ALL_EXPERTS=1`: all 384 experts sit in the tables, so the expert bytes are real.
- `DS_NO_CHECK=1`: time only.

The run used real weights from `/models/ds41-flash`, layers 0 and 2, native fp8 dense plus fp4 experts (14.9 GB uploaded), at position 3 (short context, so attention and the indexer are negligible).

| k (tokens/step) | device ms | tok/s (2 layers) | → 40 layers [A ×20] | check |
|---|---|---|---|---|
| 16 | 9.09 | 1,760 | ~88 | **FAIL**: rel err 4.1e-1 vs the emulated interpreter |
| 64 | 31.95 | 2,003 | ~100 | time only |
| 256 | 138.31 | 1,851 | ~93 | time only |

Jobs 1112-142555 and 1112-150923 (commit cd29f7a9). Every run printed 318–320 kernels, 249 of them native, and 23 int8 GEMVs with none demoted. A k=1 check with all experts was still running when this was written. It separates whether the k=16 failure comes from the multi-token native path or from the full expert tables.

Throughput is flat from k=16 to k=256, which is the GEMV signature: every token re-reads the weights. The ds-prefill session independently found ~92 tok/s for the whole model, 0.19% of SOL. It is extending the vendor cuBLASLt path to `Fp8B32`/`Mxfp4` on `claude/ds-prefill`.

What this says: today's DS device path is the decode GEMV run k times. It reaches about 1.9k tok/s for 2 layers, roughly 93 tok/s for the whole model on one GPU (which it could not hold anyway). That is ~500× below the per-GPU GEMM SOL. It is the same starvation as coder-next before #263, plus the missing MMA tiling, and steps 1–2 of the plan target exactly that.

## Reproduce

```bash
nvcc -std=c++17 -O2 -arch=native -o pgb scripts/prefill_gemm_bench.cu -lcublasLt && ./pgb 10
```

```bash
DS41_DIR=<ds dir with inference_config.json> DS_NATIVE=1 DS_ALL_EXPERTS=1 DS_STEP_K=256 DS_NO_CHECK=1 \
  cargo run --release -p tang-bench --features cuda --example ds_real_gpu
```
