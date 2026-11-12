# vLLM / SGLang tricks, mapped onto tang

2026-11-12. Static read of vllm-project/vllm, sgl-project/sglang, flashinfer-ai/flashinfer and
fla-org/flash-linear-attention (shallow clones of `main` that day; no code run). Scope: what runs
for Qwen3-Next / Qwen3-Coder-Next prefill and decode on H200, and for DeepSeek V3/V4-style MoE +
MLA + fp8/fp4. Then a "tang has it / partial / missing" mapping against tang main `a3ca706`.

The gap we're explaining (H200, same box): vLLM 0.33 FP8 coder-next prefill ~43k tok/s vs tang
~2k (3.1k with expert grouping); decode B=1 219 vs ~218; B=16 1687 vs 943 (batched-decode branch).

Paths: `vllm/` = vllm repo root, `Q:` = `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`TP/` = `vllm/third_party/flash_linear_attention/ops/`, `F/` = `vllm/model_executor/layers/fused_moe/`,
`sgl/` = `sglang/python/sglang/srt/`. Line numbers are as of the clone date.

Speedups are estimates (**[A]**), not measured. "Owner" names the tang session/area that should take it.

## Ranked table

| # | trick | where (vLLM / SGLang) | tang today | est. prefill | est. decode | effort | owner |
|---|---|---|---|---|---|---|---|
| 1 | **Big prefill batches: 8192 tokens per step** (16384 offline, 1024 seqs max). MoE weights are read once per 8k tokens, not per 256 | `vllm/engine/arg_utils.py:3038-3052` | missing — chunk 256 (prefill-sol: 30-60× below the expert-GEMM knee) | **3-6×** on its own once GEMMs are grouped | – | S (scheduler + scratch sizing), depends on #2 | prefill-gap / layerwise |
| 2 | **Token sort + block-aligned grouped expert GEMM** (`moe_align_block_size`: per-expert count → CUB exclusive scan of counts rounded up to BLOCK_M → `expert_ids[block]` → scatter; one Triton GEMM over all experts, each BLOCK_M tile owned by one expert, pad rows = sentinel `numel` masked) | `F/moe_align_block_size.py:137-147`, `csrc/moe/moe_align_sum_kernels.cu:141-230`, kernel `F/fused_moe.py` | partial — `schedule_expert_groups` sorts with an O(P²) rank (tensor_expert_group.rs), off by default in coder_derived; int8 MMA runs capped at 8 positions | 1.6× measured (grouping alone, chunk 256→2.9k); **5-10× combined with #1 and a real GEMM tile** | 1.3-2× at B≥16 | M | prefill-gap (grouped experts) + vendor-gemm |
| 3 | **Weights in fp8/bf16 on tensor cores, not GGUF int4 + int8 activation requant**. vLLM serves the FP8 checkpoint: wgmma with fp8 block-128 scales; bf16 path is Triton fused_moe | `F/oracle/fp8.py:96-139`, `F/oracle/unquantized.py:76-80` | missing for coder (GGUF Q4_K via int8 GEMV / int8 mma.sync); vendor-gemm dequants Q4K→f16 then cuBLASLt (+12.5%) | 2-3× on the GEMM share (wgmma fp8 ~1000+ TFLOP/s vs int8 mma.sync tiles at ~100s) | – at B=1 (bandwidth: Q4 is *smaller*); fp8 wins at large B | L (new weight path + loader) | vendor-gemm |
| 4 | **Chunked WY/UT GDN prefill on tensor cores** (BT=64; cumsum → kkᵀ → solve_tril (16×16 inverses merged to 64×64) → recompute w,u → state pass → o). On SM90 vLLM uses FlashInfer's CuTe-DSL warp-specialized kernel by default | `Q:187-348` (auto→flashinfer on SM90, `Q:321`); `TP/chunk.py:23-80`; `flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py` | partial — wmma C=64 WY form exists (`tensor_cuda_scan.rs`), but coder prefill runs the register-recurrent fused scan (11 ms / 138 ms chunk) | 1.1× at today's chunk; matters more once GEMMs shrink (scan becomes ~30%+) | – | M (wire `chunked_scan_plan` into coder; then WGMMA/TMA) | flash-attn / tensor-core GDN scan session |
| 5 | **FULL cuda graphs for decode, piecewise for prefill, capture sizes 1,2,4,8..248 step 8, 256..512 step 16; pad batch up** | `vllm/config/vllm.py:2355`, `vllm/config/compilation.py:708-730,855-860,926-947` | has — window graphs keyed by (t, verify, attn chunks, segs) in coder.rs:1400; batched decode needs graphs per B bucket | – | 1.2-1.5× at B=16-64 if batched path isn't graphed per bucket | S | batching |
| 6 | **Packed fused GDN decode**: one kernel reads post-conv `mixed_qkv`, fuses softplus/`g=-exp(A_log)·sp`/sigmoid β/q,k l2norm/scale, in-place state update. BK=128, **BV=4 on H200 with fp32 state**, num_warps=1, stages=3, grid (V/BV, B·HV) | `TP/fused_recurrent.py:257-505`, `Q:1796-1805` | partial — fused conv+step exists; q/k per-head norm still separate kernels (144 launches, coder-decode-gap) | – | 1.05-1.1× B=1, more at B≥16 (state traffic dominates) | S | decode |
| 7 | **Fused post-conv prep for prefill** (q/k/v split, l2norm, g, exp(g), β in one pass; chunk kernel runs with `use_qk_l2norm_in_kernel=False`) | `TP/fused_gdn_prefill_post_conv.py:164`, `Q:~1940,2069` | partial (derived graph fuses some maps) | 1.03-1.05× | – | S | prefill-gap |
| 8 | **Fused routing**: `topk_softmax` CUDA kernel (softmax|sigmoid + top-k + renormalize, one launch); DeepSeek `grouped_topk` with sigmoid + `e_score_correction_bias`; SGLang `moe_fused_gate` | `csrc/moe/topk_softmax_kernels.cu:73,157,241-251`; `F/router/grouped_topk_router.py:31,92`; `sgl/layers/moe/topk.py:1266,1960` | has (coder top-10 warp select, #262 drops softmax max/sum); DS: check bias+group path | – | ~1.02× | S | decode / ds-prefill |
| 9 | **Router weight multiplied in the down-proj GEMM epilogue** (`MUL_ROUTED_WEIGHT`), then a topk-sum reduce | `F/fused_moe.py:287,633,798`; `moe_sum` in `moe_align_sum_kernels.cu` | partial (Σw·y fused in combine) | 1.02× | 1.02× | S | prefill-gap |
| 10 | **Shared expert overlap**: second CUDA stream when tokens ≤ threshold (vLLM), or **fold the shared expert in as expert E** with fixed weight so it rides the grouped GEMM (SGLang `num_fused_shared_experts`; why E=257 configs exist). Qwen3-Next shared gate = sigmoid(1-row linear)·shared_out | `F/runner/shared_experts.py:78,125-140`; `vllm/model_executor/models/qwen3_next.py:205,225,250`; `sgl/layers/moe/mega_moe.py:249` | has for decode (parallel graph branch, coder.rs:175); prefill: separate GEMM | 1.05× (fold into grouped GEMM) | ~1.0× | S | prefill-gap |
| 11 | **Swap-AB for skinny expert tiles on Hopper** (BLOCK_M<64, BLOCK_N≥64 → compute Cᵀ so the token dim sits in the wgmma N slot) | `F/utils.py:838`, `F/fused_moe.py:357-609,846` | missing | 1.2-1.5× on expert GEMMs with m≈16-48 rows/expert | 1.2× at B=4-32 | M | vendor-gemm |
| 12 | **SiLU·mul + quant fused** (per-token or 1×128 group fp8, col-major / UE8M0 scales for DeepGEMM); rmsnorm+quant and silu+quant compile passes on for fp8 models | `F/silu_mul_token_quant.py:69`; `F/experts/deep_gemm_moe.py:286-313,573`; `vllm/config/vllm.py:185-203,327-355` | has the int8 analogue (silu·up → int8 fused); needed again for fp8 path (#3) | 1.05× | 1.03× | S (with #3) | vendor-gemm |
| 13 | **Attention: FA3 for head_dim 256, one varlen kernel for prefill+decode**; fused q-gate split + QK-RMSNorm + partial NeoX RoPE (`fused_qk_rmsnorm_rope_gate`) | `vllm/platforms/cuda.py:166-177`; `vllm/v1/attention/backends/flash_attn.py:402-405`; `vllm/model_executor/models/qwen3_next.py:371-436` | partial — FA2-style tensor-core kernel `cx_attn_fa2` + derived Hopper lowering (coder.rs:204-210); split-K for decode | 1.1× at 4k, 1.3-1.5× at 32k (12 of 48 layers) | 1.05× at 32k | M | flash-attn |
| 14 | **MTP spec decode with per-draft-token GDN state slots** (`spec_state_indices [B,k+1]`; conv update + fused recurrent take `num_accepted_tokens` and read the state of the last accepted slot — no replay) | `Q:1816-1828,1864-1869,1986-1987`; `vllm/model_executor/models/qwen3_next_mtp.py:71` | has a different design — verify windows read state, `Coder::commit` replays kept prefix; MTP sidecar + tree spec | – | replay cost: ~1.05× from slot-per-draft at small k | M | decode |
| 15 | **Async scheduling** (CPU schedules step N+1 while GPU runs N; on by default) | `vllm/config/vllm.py:1622-1649` | partial (server.rs serve loop; check host gaps in nsys) | – | 1.05-1.15× at B≥16 | S-M | serve / batching |
| 16 | **Hybrid prefix caching**: attention block size raised so one attention page ≥ one mamba-state page; mamba state checkpointed at block boundaries (`mamba_cache_mode`), FlashInfer GDN supports `state_checkpoints` | `vllm/platforms/interface.py:1088-1160`; `vllm/model_executor/models/config.py:1078-1102` | has (radix prefix cache, PR #179 prefix-routing) | (hit-rate, not kernel) | – | – | prefix-routing |
| 17 | **ba projection on an aux stream** under graphs for small token counts; SGLang fuses the GQA-interleaved qkvz/ba unpack into one Triton kernel (`fused_qkvzba_split_reshape_cat`) | `Q:1043-1090,1313-1332`; `sgl/models/qwen3_next.py:14` | has (stacked input GEMV) | – | ~1.0× | – | – |
| 18 | **Gated RMSNorm on GDN output** (`rmsnorm(x)·w·silu(z)` per head, fp32 rstd; optional fused with out_proj) | `Q:824-831,1393-1409`; `TP/layernorm_guard.py:75-173` | has (fused into merge) | – | – | – | – |
| 19 | **DeepSeek MLA**: SM90 order FLASH_ATTN_MLA → FlashMLA (64-token pages) → FlashInfer MLA → Triton; sparse MLA (V3.2/V4) FlashMLA-sparse / FlashInfer sparse SM90 (first when head 512); fp8 block-128 MoE = **Triton** at TP, FlashInfer CUTLASS at EP | `vllm/platforms/cuda.py:137-151`; `vllm/v1/attention/backends/mla/flashmla.py:57-58`; `F/oracle/fp8.py:130-139` | partial (glm5/mla.rs, deepseek/kernels.cu) | DS-specific | DS-specific | L | ds-prefill |

### Why the prefill gap is ~20×, in one paragraph

vLLM runs an 8k-token step through the FP8 checkpoint: each routed expert sees ~160 rows
(8192·10/512), so the expert GEMMs run near the ~1 PFLOP/s knee from prefill-sol §1, and each
step's whole MoE weight set (~40 GB fp8 across layers) is read once per 8k tokens. tang runs
256-token chunks of a Q4_K GGUF through int8 mma.sync with ≤8-position runs and an O(P²) sort:
each expert sees ~5 rows, the weights are re-read ~32× as often per token, and the tile math runs
far below wgmma rates. #1 + #2 + #3 are multiplicative; #4/#13 only show once those land.

## Kernel parameters worth copying

### GDN prefill (FLA Triton, vLLM vendored)
- Chunk `BT = 64` everywhere (vLLM `TP/utils.py:32`, upstream FLA, SGLang). tang's `CHUNK = 64` matches.
- `chunk_scaled_dot_kkt`: A fp32 [64,64], fp32 acc; autotune BK∈{32,64,128}, warps {2,4,8}, stages {2,3,4}.
- `solve_tril`: invert 16×16 diagonal blocks, then merge to 64×64 (`merge_16x16_to_64x64_inverse`); fp32 compute, bf16 out; TMA loads when available.
- `recompute_w_u`: BK = BV = 64.
- `chunk_gated_delta_rule_fwd_h`: grid (V/BV, N·H); BV∈{32,64}, warps {2,4}; state = up to four `[BV,64]` fp32 register tiles (K ≤ 256); per-chunk `h` saved bf16 `[B,NT,H,V,K]`, final state fp32. **SGLang pins BV=32, warps=4, stages=2** (no autotune — it writes state in place).
- `chunk_fwd_o`: BK,BV∈{64,128} (big smem), warps {2,4} on Hopper.
- Upstream FLA fuses kkt + solve_tril + w/u into one `chunk_gated_delta_rule_fwd_intra` — 3 launches → 1.
- Varlen: pack the batch as B=1 with `cu_seqlens`; precompute `chunk_indices`/`chunk_offsets`; gather initial states / scatter final states by slot.
- FlashInfer SM90: 512 threads = 4 warpgroups (WG0 TMA load/store, WG1-2 state GMMA, WG3 KK-inverse/QK), `BLK_Q = BLK_KV = 64`, D ≤ 128, stages q2 k3 v2 o2 qk2 kk2 αβ5, inverse in fp16, `exp(g)` fp32 input, fp32 state.

### GDN state / decode
- State `[H, V, K]` (K contiguous), indexed `slot·stride + hv·V·K + v·K + k`; slot 0 = null.
- vLLM state dtype = model dtype (bf16) by default, upcast per call; **SGLang defaults fp32** (and requires fp32 for its FlashInfer prefill). tang's `TANG_CODER_GDN_STATE` covers both.
- Packed decode: BK=128, BV=min(V,32), **BV=4 on H200 fp32 state**, num_warps=1, stages=3. l2norm `x/sqrt(Σx²+1e-6)`.
- Conv: `causal_conv1d_fn` BLOCK_M=8 tokens × BLOCK_N=256 channels, stages 2, SiLU fused; update kernel BLOCK_N=256.

### Fused MoE (Triton, `F/configs/`)
`E=512,N=512,device_name=NVIDIA_H200.json` (Qwen3-Next bf16, TP1):

| M | BM | BN | BK | GROUP_M | warps | stages |
|---|---|---|---|---|---|---|
| 1 | 16 | 32 | 64 | 1 | 4 | 4 |
| 16 | 16 | 128 | 64 | 1 | 4 | 4 |
| 64 | 16 | 128 | 128 | 64 | 4 | 3 |
| 256 | 16 | 128 | 128 | 1 | 8 | 3 |
| 1024 | 32 | 128 | 64 | 1 | 4 | 3 |
| 4096 | 128 | 256 | 64 | 1 | 8 | 4 |

`E=256,N=256,H200,fp8_w8a8,block_shape=[128,128]` (DeepSeek V3 TP8):

| M | BM | BN | BK | GROUP_M | warps | stages |
|---|---|---|---|---|---|---|
| 1 | 8 | 32 | 128 | 1 | 4 | 4 |
| 16 | 16 | 128 | 128 | 16 | 4 | 3 |
| 256 | 16 | 128 | 128 | 32 | 4 | 3 |
| 1024 | 64 | 128 | 128 | 32 | 4 | 4 |
| 4096 | 64 | 128 | 128 | 16 | 4 | 3 |

- fp8 block fallback (`F/fused_moe.py:1428-1462`): BM 16 (M≤64) else 64; BN 64 (M≤8) else 128; BK 128 (= quant block, scales indexed `offs_bn // 128`); GROUP_M 1 (M≤16) else 32; 4 warps.
- bf16 fallback (`:1483-1529`): BM 16/32/64/128 at M ≤32/96/512/more; BN 64 (M≤64) else 128; warps 4 (M≤128) else 8; stages 4 (M≤32) else 3.
- Align padding: `max_tokens_padded = topk·M + E·(BM−1)`; blocks = cdiv(that, BM).
- Activations: per-token-group 1×128 fp8 quant before each GEMM; weights 128×128 block scales.

### Scheduling / graphs
- `max_num_batched_tokens` 8192 (server) / 16384 (offline) on ≥70 GiB Hopper; `max_num_seqs` 1024; long-prefill threshold off.
- Graph capture sizes `[1,2,4] + range(8,256,8) + range(256,513,16)`; FULL graphs for decode-only batches, piecewise for mixed; attention/GDN cores are the split points.
- KV block 16 by default, but hybrid models raise it until an attention page ≥ a GDN state page (hundreds of tokens for Qwen3-Next).

## Not covered / follow-ups
- flashinfer trtllm-gen MoE tile configs (Blackwell; relevant on B300) and DeepGEMM grouped masked layouts — not opened.
- SGLang scheduler (chunked_prefill_size default, overlap scheduler, two-batch overlap) — not traced.
- Sampler fusion (FlashInfer top-k/top-p) and DeepSeek MTP/EAGLE path — not traced.
