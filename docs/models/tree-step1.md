# tree step 1: serving GLM-5.3-Flash on frog cloud

**Date:** 2026-10-22 · **Branch:** `frog/tree-step1` (doc-fix commit `fc46b757`) · **Scope:** read-only research; no GPU spend, no pods, no weights downloaded. Every fact cites the command that showed it. Companion fix: `docs/models/port-audit-glm53-deepseekv41.md` (H200 = 141 GB corrections, verified-facts appendix).

---

## 1. Weights

Source commands:
```
curl 'https://huggingface.co/api/models?author=zai-org&search=GLM-5'
curl 'https://huggingface.co/api/models/zai-org/GLM-5.3-Flash'          (+ -BF16, zai-org/GLM-5.3)
curl 'https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json'
curl 'https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md'
curl 'https://huggingface.co/api/models?search=GLM-5.3-Flash&sort=downloads&limit=40'
curl 'https://huggingface.co/api/models/unsloth/GLM-5.3-Flash-GGUF/tree/main/UD-Q4_K_XL'
```

- **Repo:** `zai-org/GLM-5.3-Flash`, public, 6.42M downloads, arch `Glm5NextForConditionalGeneration`, model_type `glm5_next` (text: `glm5_next_text`, vision: `glm5_next_vision`). Multimodal (image + video; 24-layer vision tower, 1024 hidden).
- **License: MIT** (README `license: mit`; HF tags `license:mit`). Commercial use OK, no attribution required (citation requested, not mandated). **The full GLM-5.3 is `license:other`** — MIT applies to the Flash only.
- **Params:** 321,323,031,390 total (~321B), ~18B active (`safetensors` field: 314.4B F8_E4M3 + 6.9B BF16 — embeds, lm_head and attention kept BF16 per `quantization_config.modules_to_not_convert`). 320B total / 18B active also stated in the README.
- **Formats and sizes on disk:**

| Format | Repo | Size on disk | Notes |
|---|---|---|---|
| FP8 E4M3 (default) | `zai-org/GLM-5.3-Flash` | **~306 GiB** (656.7 GB repo incl. duplicates; 62 shards) | dynamic activation scheme; what vLLM/SGLang recipes use |
| BF16 | `zai-org/GLM-5.3-Flash-BF16` | **~611 GiB** (642.7 GB repo) | all-BF16 321.3B params |
| GGUF Q4_K_XL | `unsloth/GLM-5.3-Flash-GGUF` (UD-Q4_K_XL) | **~227 GiB** (5 shards, ~45.4–46.6 GiB each, summed from tree API) | community; UD-IQ4_XS smaller (~190 GiB est.); Q8_0 ~335 GiB |
| GGUF misc | bartowski, vcruz305, huihui-ai (abliterated), etc. | varies | derivative/uncensored variants — avoid for frog cloud |
| FP8 variants | orcarouter/dealignai "UNCENSORED-FP8" | ~306 GiB | avoid |
| AWQ | `hell0ks/GLM-5.3-Flash-Uncensored-AWQ` | ~180 GiB est. | community, uncensored derivative |
| GPTQ/int4 | `canada-quant/GLM-5.3-Flash-W4A16-MTP` | ~160 GiB est. | community |
| NVFP4 | `RedHatAI/GLM-5.3-Flash-NVFP4` (per vLLM recipe) | ~160 GiB est. | **Blackwell only — irrelevant on H200** |

- **Full GLM-5.3 (non-Flash): released.** `zai-org/GLM-5.3` — ~753B params (751.2B FP8 + 2.1B BF16), 755.7 GB, `license:other`; also `GLM-5.3-BF16`. Not a frog cloud candidate at 2×H200.
- Context: `max_position_embeddings` 1,048,576 (1M). One MTP draft layer. Vocab 154,880.

## 2. Engine support

Source commands:
```
gh api 'repos/vllm-project/vllm/contents/vllm/model_executor/models/registry.py' | base64 -d | grep -i glm5
gh api 'repos/vllm-project/vllm/releases/tags/v0.30.0' --jq '.body'   (+ gh release list)
gh api 'repos/sgl-project/sglang/releases?per_page=8' / tags v0.5.20, v0.5.19
WebFetch https://recipes.vllm.ai/zai-org/GLM-5.3-Flash   (and sglang cookbook link from README)
```

Arch name: **`Glm5NextForConditionalGeneration`** (config.json).

### vLLM — min version **v0.30.0** (2026-09-22)
- Registry on `main`: `Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration`, `Glm5NextMTPModel` → `vllm.models.glm5next`. Support landed in #53906, **first shipped in v0.30.0** (release notes: "New models: … GLM-5.3-Flash (#53906)"). Latest release v0.32.0 (2026-10-18), which adds FlashKDA chunked prefill (1.7–3.8× faster, #55737), NoPE sparse prefill (#55738), EPLB (#55119), decode cleanups (#55736).
- **Image:** `vllm/vllm-openai:v0.32.0` (use ≥ v0.30.0; Hopper path needs **FlashInfer ≥ 0.6.18**).
- **Flags (from the official recipe):** `--tensor-parallel-size 2` (recipe shows TP4 on bigger nodes; TP2 for 2×H200), `--tool-call-parser glm47`, `--reasoning-parser glm47`, `--enable-auto-tool-choice`, `--served-model-name zai-org/GLM-5.3-Flash`; MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`.
- **Known issues/constraints (recipe + release notes):**
  - **Hopper must run BF16 KV cache** — "Hopper does not support FP8 KV cache for this model" (recipe, PD-disagg section). So no `--kv-cache-dtype fp8` on H200.
  - KV/conv-state layout env pins (`VLLM_SSM_CONV_STATE_LAYOUT=DS`, `VLLM_KV_CACHE_LAYOUT=HND`) only matter for PD disaggregation; single instance TP2 doesn't need them.
  - Sparse-MLA init errors on images with FlashInfer < 0.6.18.
  - KV offloading needs #57145 (validated on 4×H200 TP4, commit d1b4028d) — not needed for us.
- NVFP4 variant (RedHatAI) requires Blackwell; on H200 use the default FP8 checkpoint.

### SGLang — min version **v0.5.20** (2026-09-18)
- Model files exist on `main`: `python/sglang/srt/models/glm5_next.py`, `glm5_next_nextn.py` (MTP). First release mention: v0.5.20 ("GLM-5.3-Flash … #36507, #38621", cookbook link). Latest v0.5.22 (2026-10-19): DFlash decode +10% on B300, GLM-5.3-Flash breakable prefill CUDA graphs **default on** (#42845), MoE prefill autotune (#42693), DCP+MTP up to 2.1× on 8×B300.
- Cookbook: `https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash` (linked from the model README).
- Most SGLang perf work targets B300/MI355X; vLLM has the more exercised H200 path (DSA Q8KV8 FP8 sparse prefill on H200 is in SGLang #41790). **Recommend vLLM for track A.**

## 3. Memory math for 2×H200 (141 GB each, 282 GB total)

All from `curl .../zai-org/GLM-5.3-Flash/raw/main/config.json`:

- 45 layers: **34 KDA linear-attention** (`layer_types: linear_attention`, `kda_layers`) + **11 DSA sparse-MLA** (`deepseek_sparse_attention`).
- **DSA (MLA, 11 layers):** NoPE MLA — `kv_lora_rank 512`, `qk_rope_head_dim 0` (no RoPE part), so latent KV per token per layer = 512 dims. H200 → BF16 KV: **512 × 2 B = 1 KB/layer** → 11 KB... per-token DSA total = 512×2×11 = **11,264 B ≈ 11 KB**. (With a rope part it would be more; here zero.)
- **KDA (34 layers):** fixed state, no growing KV. Per stream: conv state `short_conv_kernel_size 4` × head_dim 128 × 64 heads × 2 (k,v) × 2 B = 131 KB/layer... conservative bound: 64 heads × 128 × (conv 4 + state 128) × 2 × 2 B ≈ 434 KB/layer ⇒ ~**14.8 GB per stream** at the absolute worst; with a shared/multi-query state (as in KDA implementations like Kimi's, state is per-head 128×128 bf16 → 64×128×128×2×2 = 4.2 MB/layer would be absurd — real KDA state is h×d_k×d_v where d_v is head_dim): 64 × 128 × 128 × 2 B × 34 = 70.5 MB/layer-set ≈ **70 MB conv+state per stream total** (2 B state, 4-wide bf16 conv). **Take ~70–150 MB fixed per stream for KDA** — negligible next to DSA at 128k.
- **KV per token (DSA only): 11,264 B ≈ 11 KB/token (BF16 latent, mandatory on Hopper).**
  At 128k context: 11,264 × 131,072 = **1.42 GB per stream** for DSA latent KV (+ indexer score cache: 32 heads × 128 × 128k × 2 B ≈ 1.07 GB if kept in bf16 → total **~2.5 GB/stream**; vLLM's recipe reports a 14.92M-token pool at TP4 FP8-KV on MI355X ≈ 0.55 KB/token effective after pooling/sparsity — on Hopper BF16, plan **~11–22 KB/token**, i.e. **1.4–2.8 GB per 128k stream**).
- **Weights per precision:** FP8 = **~306 GiB** (does not fit 282 GB — over budget before KV); **Q4 = ~160 GiB** at 4.25–4.5 bpw (227 GiB is unsloth's Q4_K_XL GGUF incl. BF16 attention; a Q4_K_XL serve is ~227 GiB on disk, in-GPU after load ~same).
- **Max N at 128k context:**
  - **FP8:** not viable on 2×H200 (weights alone 306 GiB > 282 GB). Needs 4×H200 TP4 or Q4.
  - **Q4 (~227 GiB weights):** free memory = 282 − 227 − ~10 (runtime/activations/MTP) ≈ **45 GB → ~16–32 concurrent 128k streams** at 1.4–2.8 GB/stream; **planning number: ~16 streams at 128k, ~30+ at 64k.**
  - If a smaller Q4 (~160 GiB pure 4-bit) is used instead: ~100 GB free → **~35–70 streams at 128k**.
- Note: KDA's 34 layers contribute **zero growing KV** — that's the whole point of the hybrid design (README: "sharply reducing long-context serving costs"). The 11 DSA layers at top-2048 sparsity also only *read* 2048 latent tokens per query; the cache still holds all 128k per stream.

## 4. Track A: vLLM on 2×H200, fronted by zest router

**Image:** `vllm/vllm-openai:v0.32.0` (FlashInfer ≥ 0.6.18 included; first supporting release v0.30.0).

**Launch command (RunPod 2×H200, FP8 checkpoint, BF16 KV):**
```bash
docker run --gpus all --shm-size 32g -p 8912:8912 \
  -v /workspace/models:/models \
  vllm/vllm-openai:v0.32.0 \
  --model /models/GLM-5.3-Flash \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8912
# BF16 KV is the Hopper default for this model; do NOT pass --kv-cache-dtype fp8.
# num_speculative_tokens 3 (recipe validates 5 on GB200; start lower on H200 TP2).
```

**Throughput estimate:** no public 2×H200 number exists for this model (public: SGLang 8×B300 decode 1199.7 tok/s single stream, #42489). Scaling by active params (~18B) vs known H200 FP8 MoE points: **single stream ~60–100 tok/s decode** (MTP3 acceptance lifts effective tok/s; agentic coding has short completions), **aggregate at 16–32 streams ~1200–2500 tok/s** on 2×H200. Prefill ~5–15k tok/s aggregate. **Treat these as planning numbers to be measured on day 1** — nothing published for H200 TP2.

**Cost:** ~$10.58/hr for the 2×H200 pod. At 2000 tok/s aggregate sustained → ~$0.19/hr per 100k output tokens → **~$0.02–0.05 per finished frog task** (40 calls, ~30k out — well inside frog api's $1.50/M out only if utilization stays high; at low utilization the floor is $10.58/hr).

**zest router integration** (read `~/Developer/zest/router/src/config.rs`, `src/proxy.rs`, `src/lib.rs`, `router.example.toml`, `~/Developer/zest/deploy/node/entrypoint.sh` — read-only):
- The router is a thin **OpenAI `/v1/chat/completions` reverse proxy**: it verifies HMAC, admits by tier (`tiers.<name>.max_streams/max_tokens`), forwards the body to `upstream` (or a `[[models]]` entry matched by the request's `model` id) at `POST {upstream}/v1/chat/completions`, rewrites `model` in responses/SSE to the public name, records usage, and serves `/v1/models`, `/healthz` (+`?deep=1` 1-token probe), `/metrics`. `config.rs:77 route()` matches `[[models]] id` → `upstream`; `proxy.rs:108-125` forwards.
- **A `frog-tree` model id needs exactly one `[[models]]` block:**
  ```toml
  [[models]]
  id = "frog-tree"          # what clients send
  upstream = "http://127.0.0.1:8912"   # the vLLM server above
  ctx = 131072              # metrics label
  kv_slots = 32             # metrics label (max-num-seqs)
  ```
  placed in `router.toml` alongside the default tang upstream. Requests with `"model": "frog-tree"` route to vLLM; everything else keeps going to tang-llm.
- **Verdict: the router can front a vLLM/SGLang OpenAI backend as-is** for `/v1/chat/completions` + `/v1/models` + `/healthz?deep=1`. Caveats:
  1. `upstream_key` — one key per router; vLLM would share the tang upstream key (fine) — set `--api-key` on vLLM to `ZEST_UPSTREAM_KEY`.
  2. The entrypoint script (`entrypoint.sh`) is tang-llm-specific — it starts `tang-llm serve`, not vLLM. Track A needs a new node entrypoint (or just run the router container with a hand-written router.toml on the pod) — a zest-repo change owned by zest, not tang.
  3. The router parses usage from tang-llm's response shape (`proxy.rs:266`); vLLM's OpenAI `usage` object is standard, but SSE accounting should be smoke-tested — verify `usage` fields land in the spool before relying on billing.

## 5. Track B: tang port — ordered plan (corrected numbers)

Base: the corrected audit (`docs/models/port-audit-glm53-deepseekv41.md`, commit `fc46b757`). Corrections that change the plan: expert FFN is **2048** (not 640 — new kernel geometry after all), hidden **4096**, vocab **154,880**, DSA is **NoPE MLA** (no RoPE), and **2×H200 fits Q4 without offload** — offload is off the critical path for bring-up.

Ordered:
1. **Config/ GGUF parse (few days):** `Arch::Glm5Next` in `flash/reference.rs`; GGUF tensor-name mapping for the 62-shard FP8 HF layout is a non-starter (tang reads GGUF, not safetensors) — parse unsloth UD-Q4_K_XL; KDA/MLA/mHC tensor names need a dump-and-map pass.
2. **mHC Sinkhorn (1 week, shared with DSv4.1):** `hc_sinkhorn` (4 streams, 20 iters, `hc_eps 1e-6`); do it first — shared, and it gates everything downstream.
3. **MLA NoPE (2–3 weeks):** latent KV (kv_lora 512, no rope part — *simpler* than DeepSeek MLA), absorbed projections, q_lora 1536, qk_nope 256, v_head 256; pooled-key indexer (32 heads, kpool 4, top-2048) after core MLA.
4. **KDA (1–2 weeks):** extend GDN kernel with vector gate (per-channel decay, `gate_lower_bound -5`, 64×128, short conv k=4).
5. **MoE (1–2 weeks):** sigmoid router + `noaux_tc` bias, top-8 of 288, routed scale 2.5, shared expert, **new 2048 expert-FFN geometry**, swiglu_limit 10.
6. **MTP (few days):** 1 nextn layer; wire into existing flash spec-decode.
7. **E2E + bench (1 week):** `flash-ref`/`flash-tcheck` parity, then tok/s at 128k on 2×H200 Q4 (fits, no offload) and the 1×H200 offload path after.

**Duration: ~6–9 weeks** (audit's 4–6 weeks + the expert-FFN geometry it missed + GGUF mapping). Memory corrections make it *easier*: no offload needed for 2×H200 bring-up, and 1×H200 (141 GB) offload is a stretch goal, not a prerequisite.

## Summary table

| Item | Value |
|---|---|
| Weights | `zai-org/GLM-5.3-Flash`, **MIT**, 321B total / ~18B active; full GLM-5.3 also released (`license:other`, ~753B) |
| Quants | FP8 ~306 GiB (official), BF16 ~611 GiB (official), GGUF Q4_K_XL ~227 GiB (unsloth), community AWQ/GPTQ/NVFP4 (Blackwell-only) |
| vLLM | ≥ **v0.30.0**, image `vllm/vllm-openai:v0.32.0`; TP2 + glm47 parsers + MTP spec; **BF16 KV on Hopper** |
| SGLang | ≥ **v0.5.20** (v0.5.22 best); H200 path less exercised than vLLM's |
| Memory 2×H200 | DSA-only KV ≈ 11 KB/tok bf16 (~2.5 GB/128k stream); FP8 weights don't fit; **Q4 → ~16–32 streams at 128k** |
| Track A | vLLM TP2 on 2×H200, ~60–100 tok/s single / ~1.2–2.5k tok/s aggregate (to be measured), **$10.58/hr** |
| Router | Fronts vLLM **as-is** via one `[[models]] id="frog-tree"` block; needs a zest-side entrypoint + usage-shape smoke test |
| Track B | **~6–9 weeks**: GGUF map → Sinkhorn HC → MLA NoPE → KDA → MoE(2048 FFN) → MTP → E2E; 2×H200 Q4 needs no offload |
