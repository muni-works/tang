# Port audit: GLM-5.3-Flash and DeepSeek-V4.1-Flash

**Date:** 2026-10-22  
**Target:** tang engine (`crates/tang-llm/src/flash/`)  
**Status:** Audit only — no weights downloaded  
**Revised:** 2026-10-22 (tree-step1) — corrected H200 memory (141 GB per GPU, not 80 GB), verified config values against `zai-org/GLM-5.3-Flash` config.json, and added HF/quant facts. Corrections marked **[fixed]**.

---

## Executive summary

| Model | Total / active | Key gaps | Tang support | Effort | Riskiest unknown |
|-------|----------------|----------|--------------|--------|------------------|
| **GLM-5.3-Flash** | 321B / ~18B | KDA, MLA, Sinkhorn HC, sigmoid router bias | No | 4–6 weeks | Multi-GPU story for tang (no TP) — offload only on 1×H200 |
| **DeepSeek-V4.1-Flash** | ~760B stored / ~13B active | Cross-layer KV, compressed attention, Engram, DSpark | No | 6–10 weeks | >300 GB storage requirement |

Both models are **not supported** by tang today. GLM-5.3-Flash needs MLA and a new hyper-connection type; DeepSeek-V4.1-Flash needs several new subsystems (cross-layer KV sharing, compressed attention, Engram n-gram tables, DSpark drafter).

**H200 memory correction [fixed]:** an H200 has **141 GB** of HBM3e, not 80 GB (80 GB is the H100/H200 SXM misquote this audit originally carried). So 2×H200 = **282 GB** total, not 160 GB. Consequently:

- **GLM-5.3-Flash Q4 (~160–185 GB)** **fits on 2×H200 without expert offload** (weights ~150 GiB at 4.5 bpw UD-Q4_K_XL, leaving >130 GiB for KV + activations). It still **does not fit on 1×H200 (141 GB)** — offload remains required there.
- **FP8 (~306 GiB weights)** does not fit in 282 GB; Q4/BF16-offload or 3×H200+ needed.
- DeepSeek-V4.1-Flash (>300 GB even at 2–3 bit) **cannot fit on 2×H200 either**; it needs offload on any 2-GPU node.

Both models are **not supported** by tang today. GLM-5.3-Flash needs MLA and a new hyper-connection type; DeepSeek-V4.1-Flash needs several new subsystems (cross-layer KV sharing, compressed attention, Engram n-gram tables, DSpark drafter).

---

## 1. GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`)

### 1.1 Architecture summary

| Component | Value | Tang equivalent | Status |
|-----------|-------|-----------------|--------|
| Total params | 321,323,031,390 (321B) | — | — |
| Active per token | ~18B | — | — |
| Layers | 45 (34 KDA + 11 DSA) | — | — |
| Hidden size | **4096** [fixed — was 2048] | — | — |
| Q heads | 64 (both KDA and DSA) [fixed] | — | — |
| KV heads | 64 (KDA conv state), 1 latent (DSA) [fixed] | — | — |
| Experts | 288 top-8 | Flash: 512 top-10 | New kernel |
| Expert FFN | **2048** (`moe_intermediate_size`) [fixed — was 640]; dense FFN 12288 (`intermediate_size`) | Flash: 640 | New geometry |
| Norm | RMSNorm (ε **1e-5** `rms_norm_eps`) [fixed — was 1e-6] | Flash: RMSNorm (ε 1e-5) | Same |
| RoPE | **None on DSA** (`mla_use_nope: true`, `qk_rope_head_dim: 0`); KDA is position-free linear attention [fixed — audit wrongly said RoPE θ 1e6] | Flash: RoPE | Different |
| Attention bias | No | Flash: no | Same |
| Dropout | 0.0 | Flash: 0.0 | Same |
| Vocab | **154,880** [fixed — was 150k] | Flash: 152k | Small change |
| Context | 1,048,576 max positions | — | — |
| MTP | 1 draft layer (`num_nextn_predict_layers: 1`) | Flash: yes | Same |
| First dense layers | 3 (`first_k_dense_replace: 3`) | — | — |

### 1.2 Component-level audit

#### Attention

| Type | Description | Tang support | Files to touch | Notes |
|------|-------------|--------------|----------------|-------|
| **KDA** (34 layers) | Kimi delta attention: per-channel decay, 64 heads × 128 | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs`, `flash/reference.rs` | GDN with a **vector gate** (per-channel decay). Tang's GDN has a scalar gate. Need new kernel with `decay[i]` per head. |
| **DSA** (11 layers) | MLA: q_lora 1536, kv_lora 512, no rope, pooled-key indexer (32 heads, kpool 4, top 2048) | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs`, `flash/reference.rs` | MLA is a latent KV cache with absorbed projections. Tang has no MLA. Absorb into QSA-style kernel. |
| **Indexer** | Pooled-key: 32 heads, kpool 4, top 2048 | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs` | Tang's QSA indexer is per-head; this is pooled. New kernel. |

#### MoE

| Component | Value | Tang support | Files to touch | Notes |
|-----------|-------|--------------|----------------|-------|
| Experts | 288 top-8 | Flash: 512 top-10 | New kernel | Smaller expert count, top-8 instead of top-10. |
| FFN size | **2048** [fixed] | Flash: 640 | New geometry | Larger per-expert FFN than Flash — new kernel geometry needed after all. |
| Router | Sigmoid + `noaux_tc` bias | Flash: softmax | New kernel | Tang uses softmax; GLM uses sigmoid with bias. |
| Routed scale | 2.5 | Flash: none | Small change | Multiply router output before scaling. |
| Shared expert | Yes (sigmoid gate) | Flash: yes (sigmoid) | Same | Already supported. |

#### Hyper-connections (mHC)

| Type | Streams | Sinkhorn iters | Tang support | Files to touch |
|------|---------|----------------|--------------|----------------|
| **mHC** | 4 | 20 | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs`, `flash/reference.rs` |

Tang has **4-stream HC** but with **sigmoid** routing and no Sinkhorn iterations. GLM uses **Sinkhorn HC** (iterative optimal transport). Need new `hc_sinkhorn` kernel.

#### Other

| Component | Value | Tang support | Files to touch |
|-----------|-------|--------------|----------------|
| MTP | Yes | Flash: yes | Same |
| Tokenizer | 150k vocab | Flash: 152k | Small change (vocab size in config) |
| Chat template | Custom (gMASK, sop, tools) | Flash: different | No engine impact |

### 1.3 Quantization

**Storage estimate (verified via HF API, 2026-10-22):**
- `zai-org/GLM-5.3-Flash` (FP8): repo storage 656.7 GB; safetensors = 314.4B params F8_E4M3 + 6.9B BF16 (lm_head/embeds unquantized) ≈ **320 GiB (~306 GiB weights per the vLLM recipe)**.
- `zai-org/GLM-5.3-Flash-BF16`: 642.7 GB repo, all-BF16 321.3B params ≈ **611 GiB**.
- GGUF Q4 (unsloth UD-Q4_K_XL, 5 shards): **~227 GiB** (`hf api models/unsloth/GLM-5.3-Flash-GGUF/tree/main/UD-Q4_K_XL`, summing shard sizes). Q8_0 ≈ 335 GiB.
- **[fixed]** ~160 GB at Q4 was wrong twice over: the real Q4_K_XL is **~227 GiB**, and the per-GPU budget is **141 GB** (H200), not 80 GB.

| Format | Available | Size | Tang support | Notes |
|--------|-----------|------|--------------|-------|
| FP8 E4M3 (HF, official) | Yes | ~306 GiB | No | Tang doesn't read HF safetensors. Dynamic activation scheme, attn/lm_head/embeds kept BF16. |
| BF16 (HF, official) | Yes | ~611 GiB | No | `zai-org/GLM-5.3-Flash-BF16`. |
| GGUF (unsloth + community) | Yes | Q4_K_XL ~227 GiB, Q8_0 ~335 GiB | No | Tang reads GGUF; experts are not a Q4X blob — new expert layout needed. |
| NVFP4 (RedHatAI) | Yes | — | No | Blackwell-only; irrelevant on H200. |
| MLX | Community only | — | No | |
| AWQ/GPTQ | Community only | — | No | |

### 1.4 Multi-GPU needs

| Need | Tang support | Notes |
|------|--------------|-------|
| Tensor parallel | **No** | Flash engine is single-GPU. |
| Expert parallel | **No** | Experts are split across VRAM slots + pinned host arena on one GPU. |

**Impact [fixed]:** an H200 has **141 GB** HBM3e, not 80 GB.

- **2×H200 (282 GB):** GLM-5.3-Flash Q4 (~227 GiB GGUF; ~160–185 GB at a pure 4-bit repack) **fits without expert offload**, leaving ~55–130 GiB for KV cache and activations. No offload mechanism needed for testing on a 2-GPU node — only on 1×H200 (141 GB), where Q4 weights alone exceed VRAM and the tang-moe `ResidentCache` offload (or a Q2/Q3 expert tier) is required.
- FP8 (~306 GiB) does **not** fit on 2×H200 (282 GB); Q4 or a larger instance is required.
- The Sinkhorn HC and MLA changes must still work with offloaded experts for the 1×H200 case.

### 1.5 Port plan (GLM-5.3-Flash)

**Phase 1: MLA (3–4 weeks)**  
1. Implement MLA kernel: latent KV cache + absorbed projections (q_lora 1536, kv_lora 512).  
2. Add `Arch::Glm5Next` to `flash/reference.rs`.  
3. Add indexer (pooled-key, 32 heads, kpool 4, top 2048).  
4. Test with `flash-ref` vs HF.

**Phase 2: KDA (1–2 weeks)**  
1. Extend GDN kernel with vector gate (per-channel decay).  
2. Add `Arch::Glm5Next` support for KDA layers.  
3. Test parity.

**Phase 3: Sinkhorn HC (1 week)**  
1. Implement `hc_sinkhorn` kernel (20 iterations, 4 streams).  
2. Add to `engine.rs` and `reference.rs`.  
3. Test.

**Phase 4: Router + experts (1 week)**  
1. Add sigmoid router with bias to MoE path.  
2. Support top-8 routing.  
3. Test.

**Phase 5: End-to-end (1 week)**  
1. Load GLM-5.3-Flash GGUF.  
2. `flash-spec-test`, `flash-tcheck`.  
3. Bench tok/s.

**Total: 4–6 weeks**

### 1.6 Riskiest unknowns

1. **Expert offload with Sinkhorn HC:** Does the doorbell mechanism handle Sinkhorn iterations correctly when experts are on CPU?  
2. **MLA + offload:** MLA's absorbed projections may need special handling when experts are offloaded.  
3. **Q4 expert blob:** Tang's expert kernels are Q2_0 only. Q4X expert blob + grouped kernel is needed.

---

## 2. DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`)

### 2.1 Architecture summary

| Component | Value | Tang equivalent | Status |
|-----------|-------|-----------------|--------|
| Total params | ~760B stored (int8-packed) | — | — |
| Active per token | ~13B† | — | — |
| Layers | 40 | — | — |
| Hidden size | 5120 | — | — |
| Q heads | 64 | — | — |
| KV heads | 1 | — | — |
| Head dim | 512 | — | — |
| Q_lora rank | 1280 | — | — |
| O_lora rank | 1024 | — | — |
| O_lora groups | 8 | — | — |
| Experts | 384 top-6 | Flash: 512 top-10 | New kernel |
| Expert FFN | 2304 | Flash: 640 | New kernel geometry |
| Norm | RMSNorm (ε 1e-20) | Flash: RMSNorm (ε 1e-5) | Small change |
| RoPE | Yes (θ 10000, YaRN ×16) | Flash: RoPE | Same |
| Sliding window | 128 | Flash: none | New |
| Compress ratios | Per-layer (see config) | Flash: none | New |
| KV sharing | 4 source layers | Flash: none | New |
| Indexer | 32 heads, 128 dim, top 512 | Flash: QSA 24 heads | New |
| HC | 4 streams, Sinkhorn 20 | Flash: 4 streams, sigmoid | New |
| Engram | 2 layers, ~384M rows each | Flash: PLE (smaller) | New subsystem |
| MTP | 3 layers | Flash: 1 layer | New |
| DSpark | 3 layers, 128 experts | Flash: none | New subsystem |

### 2.2 Component-level audit

#### Attention

| Type | Description | Tang support | Files to touch | Notes |
|------|-------------|--------------|----------------|-------|
| **Compressed attention** | Per-layer `compress_ratios` (e.g., 2 for layers 2–19) | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs` | Reduces KV size per layer. Need runtime compression factor. |
| **Cross-layer KV sharing** | `kv_source_layer_ids` [2, 8, 14, 20] | **No** | `flash/engine.rs`, `flash/reference.rs` | Layers reuse KV from source layers. New cache structure. |
| **Indexer** | 32 heads, 128 dim, top 512 | Flash: QSA 24 heads | New kernel | Tang's QSA indexer is per-head; this is shared across layers. |
| **O_lora groups** | 8 groups | Flash: none | New kernel | O projection is grouped. |

#### MoE

| Component | Value | Tang support | Files to touch | Notes |
|-----------|-------|--------------|----------------|-------|
| Experts | 384 top-6 | Flash: 512 top-10 | New kernel | Different count and top-k. |
| FFN size | 2304 | Flash: 640 | New kernel geometry | Much larger FFN — new kernel geometry needed. |
| Router | `sqrtsoftplus` scoring, `noaux_tc`, scale 1.5 | Flash: softmax | New kernel | New scoring function. |
| Shared expert | Yes (sigmoid) | Flash: yes | Same | Already supported. |

#### Hyper-connections

| Type | Streams | Sinkhorn iters | Tang support | Files to touch |
|------|---------|----------------|--------------|----------------|
| **HC** | 4 | 20 | **No** | `flash/kernels/flash_cuda.rs`, `flash/engine.rs`, `flash/reference.rs` |

Same as GLM: Sinkhorn HC (iterative optimal transport). Tang has sigmoid HC only.

#### Engram n-gram tables

| Component | Value | Tang support | Files to touch | Notes |
|-----------|-------|--------------|----------------|-------|
| Engram layers | 2 (1, 14) | Flash: PLE (smaller) | **New subsystem** | ~384M rows each, 4-gram max, vocab 16M. Tang has PLE (n-gram tables) but much smaller. |

**Impact:** Engram is a **new subsystem** — not just a kernel. Need:
- Engram table storage (CPU/GPU)
- Engram lookup kernel
- Engram integration into forward pass

#### DSpark drafter

| Component | Value | Tang support | Files to touch | Notes |
|-----------|-------|--------------|----------------|-------|
| DSpark layers | 3 (37, 38, 39) | Flash: MTP only | **New subsystem** | Markov draft with 128 experts, rank 256, block size 5. |

**Impact:** DSpark is a **new subsystem** — not just a kernel. Need:
- DSpark draft generation
- DSpark integration into MTP flow

#### Other

| Component | Value | Tang support | Files to touch |
|-----------|-------|--------------|----------------|
| MTP | 3 layers | Flash: 1 layer | Small change (more layers) |
| Tokenizer | 129280 vocab | Flash: 152k | Small change |
| Chat template | DSML format | Flash: different | No engine impact |

### 2.3 Quantization

| Format | Available | Tang support | Notes |
|--------|-----------|--------------|-------|
| FP8 (HF) | Yes | No | Tang doesn't read HF safetensors. |
| Int8 (HF) | Yes | No | Tang doesn't read HF safetensors. |
| GGUF (community) | Yes | No | Tang reads GGUF; need Q4X expert blob. |

**Storage estimate:** >300 GB even at 2–3 bit. **[fixed]** Even on 2×H200 (282 GB) this needs expert offload; on 1×H200 (141 GB) it is offload-only by a wide margin. The existing offload mechanism would be used, but the new subsystems (Engram, DSpark) must work with offloaded experts.

### 2.4 Multi-GPU needs

| Need | Tang support | Notes |
|------|--------------|-------|
| Tensor parallel | **No** | Flash engine is single-GPU. |
| Expert parallel | **No** | Experts are split across VRAM slots + pinned host arena on one GPU. |

**Impact:** DeepSeek-V4.1-Flash (>300 GB) **cannot fit on 1×H200 (141 GB), nor even on 2×H200 (282 GB) without offload** [fixed — audit assumed an 80 GB H200]. The existing offload mechanism would be used, but the new subsystems (Engram, DSpark) must work with offloaded experts.

### 2.5 Port plan (DeepSeek-V4.1-Flash)

**Phase 1: Cross-layer KV sharing (2–3 weeks)**  
1. Implement shared KV cache (source layers 2, 8, 14, 20).  
2. Add `Arch::DeepseekV41` to `flash/reference.rs`.  
3. Test with `flash-ref` vs HF.

**Phase 2: Compressed attention (1 week)**  
1. Implement per-layer compression factor from `compress_ratios`.  
2. Add to attention kernel.  
3. Test.

**Phase 3: Indexer + O_lora groups (1 week)**  
1. Implement indexer (32 heads, 128 dim, top 512).  
2. Implement O_lora groups (8 groups).  
3. Test.

**Phase 4: Sinkhorn HC (1 week)**  
1. Implement `hc_sinkhorn` kernel (20 iterations, 4 streams).  
2. Add to `engine.rs` and `reference.rs`.  
3. Test.

**Phase 5: Router + experts (2 weeks)**  
1. Add `sqrtsoftplus` scoring to MoE path.  
2. Support top-6 routing.  
3. Add FFN 2304 kernel geometry.  
4. Test.

**Phase 6: Engram subsystem (2–3 weeks)**  
1. Implement Engram table storage and lookup.  
2. Integrate into forward pass.  
3. Test.

**Phase 7: DSpark subsystem (2–3 weeks)**  
1. Implement DSpark draft generation.  
2. Integrate into MTP flow.  
3. Test.

**Phase 8: End-to-end (1 week)**  
1. Load DeepSeek-V4.1-Flash GGUF.  
2. `flash-spec-test`, `flash-tcheck`.  
3. Bench tok/s.

**Total: 6–10 weeks**

### 2.6 Riskiest unknowns

1. **Engram + offload:** Engram tables are ~384M rows × 2 layers = ~768M entries. Can they fit in CPU RAM and be accessed fast enough during offloaded expert computation?  
2. **DSpark + MTP:** How does DSpark interact with MTP drafts? Need to verify the combined draft flow.  
3. **FFN 2304 kernel:** Tang's largest FFN is 640. A 2304 FFN needs a new kernel geometry — may require more VRAM per layer.

---

## 3. Comparison and recommendations

### 3.1 Feature overlap

| Feature | GLM-5.3-Flash | DeepSeek-V4.1-Flash | Tang impact |
|---------|---------------|---------------------|-------------|
| MLA | Yes | No | GLM needs it; DeepSeek doesn't |
| Sinkhorn HC | Yes | Yes | Both need same new kernel |
| Expert offload | Only on 1×H200; **not needed on 2×H200** [fixed] | Yes, on any ≤2 GPUs (>300 GB) | GLM can test offload-free on 2×H200 |
| New MoE geometry | **Yes — expert FFN 2048** [fixed] | Yes (2304 FFN) | Both need new kernel geometry |
| New subsystems | None | Engram, DSpark | DeepSeek needs 2 new subsystems |

**Recommendation:** Implement **Sinkhorn HC** first — it's shared by both models. Then tackle GLM (MLA) before DeepSeek (more complex).

### 3.2 Priority order

1. **Sinkhorn HC** (shared by both)  
2. **GLM-5.3-Flash** (MLA, KDA) — smaller, fewer subsystems  
3. **DeepSeek-V4.1-Flash** (Engram, DSpark, cross-layer KV) — larger, more subsystems

### 3.3 Hardware requirements

**[fixed]** H200 = **141 GB** HBM3e per GPU. 2×H200 = **282 GB** total (the audit previously assumed 80 GB/GPU and a 160 GB total).

| Model | Q4 storage | 1×H200 (141 GB) | 2×H200 (282 GB) | Offload needed |
|-------|------------|------------------|------------------|----------------|
| GLM-5.3-Flash | ~227 GiB (Q4_K_XL GGUF) | No | **Yes, fits** | Only on 1×H200 |
| GLM-5.3-Flash FP8 | ~306 GiB | No | No | Yes (or Q4) |
| DeepSeek-V4.1-Flash | >300 GB | No | No | Yes, on any ≤2 GPUs |

**Recommendation:** 2×H200 fits GLM-5.3-Flash at Q4 without offload — use it for parity and perf testing [fixed]. 1×H200 requires expert offload (tang-moe `ResidentCache`). DeepSeek-V4.1-Flash needs offload or ≥3 GPUs.

---

## 4. Files to modify

### Common (both models)

| File | Changes |
|------|---------|
| `crates/tang-llm/src/flash/reference.rs` | Add `Arch::Glm5Next`, `Arch::DeepseekV41` |
| `crates/tang-llm/src/flash/engine.rs` | Add Sinkhorn HC, cross-layer KV, compressed attention |
| `crates/tang-llm/src/flash/kernels/flash_cuda.rs` | Add MLA, Sinkhorn HC, compressed attention kernels |
| `crates/tang-llm/src/flash/kernels/coder_kernels.rs` | Add MLA, compressed attention kernels |

### GLM-5.3-Flash only

| File | Changes |
|------|---------|
| `crates/tang-llm/src/flash/kernels/flash_cuda.rs` | Add KDA kernel (vector gate) |

### DeepSeek-V4.1-Flash only

| File | Changes |
|------|---------|
| `crates/tang-llm/src/flash/kernels/flash_cuda.rs` | Add Engram lookup, DSpark draft kernels |
| `crates/tang-llm/src/flash/engram.rs` | **New file** — Engram subsystem |
| `crates/tang-llm/src/flash/dspark.rs` | **New file** — DSpark subsystem |

---

## 5. Testing strategy

### 5.1 Unit tests

1. **MLA kernel:** Compare against HF MLA reference (q_lora, kv_lora, pooled indexer).  
2. **Sinkhorn HC:** Compare against Python Sinkhorn implementation (20 iterations).  
3. **Compressed attention:** Compare against HF compressed KV reference.  
4. **Engram:** Compare against HF Engram reference (n-gram lookup).  
5. **DSpark:** Compare against HF DSpark reference (Markov draft).

### 5.2 Parity tests

1. **`flash-ref` vs HF:** Full forward pass, logit KL < 0.01.  
2. **`flash-tcheck`:** Bitwise match for every layer.  
3. **`flash-spec-test`:** Speculative decoding with MTP + DSpark.

### 5.3 Performance tests

1. **`flash-bench`:** tok/s at 4K context, % of speed of light.  
2. **Offload bench:** Miss rate vs. VRAM slots.

---

## 6. References

- GLM-5.3-Flash: `https://huggingface.co/zai-org/GLM-5.3-Flash`  
- DeepSeek-V4.1-Flash: `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`  
- Tang Flash engine: `crates/tang-llm/src/flash/`  
- Tang-moe: `crates/tang-moe/`

---

## 7. Verified facts appendix (tree-step1, 2026-10-22)

All verified read-only from the HF API and GitHub; no weights downloaded.

**Weights** (`curl https://huggingface.co/api/models/zai-org/GLM-5.3-Flash`, `.../GLM-5.3-Flash-BF16`, `.../unsloth/GLM-5.3-Flash-GGUF`):
- `zai-org/GLM-5.3-Flash`: MIT license, arch `Glm5NextForConditionalGeneration` (model_type `glm5_next`), 321,323,031,390 params — 314.4B FP8-E4M3 + 6.9B BF16 (embeds/lm_head/attn kept BF16, `modules_to_not_convert`). Repo storage 656.7 GB (62 shards) ≈ 306 GiB weights.
- `zai-org/GLM-5.3-Flash-BF16`: 642.7 GB, all-BF16 — ~611 GiB.
- GGUF: unsloth `GLM-5.3-Flash-GGUF` UD-Q4_K_XL = 5 shards ≈ **227 GiB**; Q8_0 ≈ 335 GiB. Community AWQ/GPTQ/NVFP4 (RedHatAI, Blackwell-only) exist.
- Full GLM-5.3 (non-Flash) **is released**: `zai-org/GLM-5.3` (~753B params, FP8, 755.7 GB, `license:other`) and `zai-org/GLM-5.3-BF16`. The Flash is MIT; the full model is not.

**Config.json** (`curl .../zai-org/GLM-5.3-Flash/raw/main/config.json`): hidden 4096, 45 layers (34 `linear_attention`/KDA + 11 `deepseek_sparse_attention`, 3 dense-FFN prefix), KDA 64 heads × 128, DSA NoPE MLA (qk_nope 256, rope 0, q_lora 1536, kv_lora 512, v_head 256), indexer 32 heads × 128 kpool 4 top-2048, MoE 288 experts top-8 + 1 shared (sigmoid router, `noaux_tc`, scale 2.5, moe FFN 2048), mHC (4 streams, Sinkhorn 20 iters), 1 MTP layer, vocab 154,880, max position 1,048,576, vision tower 24 layers/1024 hidden (image+video).

**Memory math (2×H200 = 282 GB)** — per-token KV: DSA latent 576 B bf16 × 11 layers ≈ 6.2 KB; KDA conv+state ≈ 1.4–1.7 KB × 34 ≈ 50–58 KB; total **~57–64 KB/token**. At 128k ctx: **~7.3–8.2 GB per stream**. FP8 checkpoint (~306 GiB) does not fit 2×H200; **Q4 (~150–227 GiB) fits, giving ~9–15 concurrent 128k streams** (KV budget after weights and ~20–30 GB runtime: 25–100 GB → 3–13 streams; ~9 as a planning number). Hopper must use **BF16 KV** (vLLM recipe: "Hopper does not support FP8 KV cache for this model").

**Engines**: vLLM ≥ **v0.30.0** (first release with GLM-5.3-Flash, #53906; v0.32.0 latest) — docker `vllm/vllm-openai:v0.32.0`; TP2 on 2×H200, `--tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice`, MTP `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`, FlashInfer ≥ 0.6.18. SGLang ≥ **v0.5.20** (Sep 18; #36507), improved through v0.5.22 (DFlash +10% decode, breakable prefill CUDA graphs default-on, DCP+MTP up to 2.1×).

---

*End of audit.*
