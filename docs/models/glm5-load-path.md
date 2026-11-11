# GLM-5.3-Flash: real-model load path (UD-Q4_K_XL, 6 shards, ~244 GB)

How the first token run loads `unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL on a 2×H200 pod:
mmap only, never a whole-file read; the split picks per-layer byte budgets from the shard
headers; first tokens come from `glm5-ref` (CPU reference, exact) and then the CUDA engine
once the glm5 kernels are wired in.

## What already exists (no work needed)

- **`Gguf::open` is mmap-only** (`crates/tang-llm/src/gguf.rs`): `Mmap::map` per shard,
  the header parsed from the mapping, tensor data faulted in page-by-page only when a
  dequantiser walks it. The whole 244 GB is *never* read into RAM; RSS stays at header +
  whatever the current layer touched. `discover_shards` handles unsloth's
  `...-00001-of-00006.gguf` naming (test `gguf::tests::shard_names`); shard 1 of the real
  file is 9.4 MB of metadata + 0 tensors and `Gguf::open` takes metadata from the first
  shard and tensors from every shard, exactly matching that layout.
- **`moe::MoeParams::from_gguf` keeps experts as GGUF handles** (`W::Gguf(g, info)`), and
  `W::mat` dequantises one expert at a time (`g.dequantize`), so the 190 GB of routed
  experts is also streamed per layer, not resident. The fp32 reference forward never holds
  more than one layer's weights (~4.4 GB worst, the MoE layers).
- **Per-layer byte budget** for the split is already in `docs/models/glm5-gguf-map.md`
  (measured from the shard headers): routed experts 4.379 GB/sparse layer, everything else
  per family listed there; totals ~186 GiB of payload.
- **Split point** (`coder::split_point` / `balanced`) sums per-layer bytes from the tensor
  table and picks the balanced cut — for glm5 it needs `hp.n_layer` and the expert tensor
  names, which match; at the real geometry it lands at k=23 (92.3/93.6 GiB), same as the
  plan in `docs/models/glm5-split-plan.md`.

## The one gap: RAM ceiling of the fp32 reference on the real model

`glm5-ref` dequantises per layer and frees between layers, but the **MoE layers are the
risk**: W5's `MoeParams` keeps routed experts as GGUF handles and dequantises per expert
inside `route_and_run` — good — but `KdaLayerW`/`MlaWeights` load all their layer's
attention matrices up front (a few hundred MB, fine). The dense `Mat::apply` path materialises
`[t_len][rows]` outputs — at prefill t=52 that's noise; at serve-scale prefill (4096-token
chunks) it's 4096 × 8192 × 4 B = 134 MB per matmul, fine.

So `glm5-ref` runs the real model in well under 8 GB RSS. It is slow (f32 matmuls on CPU,
~0.5 tok/s single-threaded at hidden 4096) — that is what it is for: exactness, not speed.

## Pod launch commands (RunPod 2×H200)

```bash
# pod: 2×H200, 1.9 TB NVMe, container with CUDA 12.x + rust
git clone https://github.com/muni-works/tang tang && cd tang
cargo build --release -p tang-llm --no-default-features --features cuda
# (~6 min from a warm target/ on the persistent volume /root/tang)

# 1. fetch the shards (only headers needed for the split plan; full files for the run)
pip install -U huggingface_hub
huggingface-cli download unsloth/GLM-5.3-Flash-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir /root/glm53

# 2. CPU reference first tokens (exactness gate; runs on CPU, no GPU needed)
target/release/tang-llm glm5-ref /root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --ids-file /root/prompt.ids --top 8 --logits /root/first.logits -v
# shard discovery is automatic; RSS stays < 8 GB; ~0.5 tok/s at hidden 4096 on the pod CPU

# 3. GPU parity on the tiny GGUF (W0's fixture copied to /private/tmp/glm5-tiny on the pod,
#    the path the tests read): W2–W5 kernel tests + the engine's GPU-FFN vs CPU check
rsync -a <mac>:/private/tmp/glm5-tiny/ /private/tmp/glm5-tiny/
TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --no-default-features --features cuda \
    --lib glm5 -- --nocapture
TANG_REQUIRE_CUDA=1 TANG_GPUS=2 TANG_SPLIT_AT=2 cargo test --release -p tang-llm \
    --no-default-features --features cuda --lib glm5::engine -- --nocapture

# 4. first tokens on the real model: one GPU can't hold the experts (190 GB), so start split.
#    TANG_GLM5_CACHE=1 keeps the f32 attention weights in RAM (~26 GB) instead of
#    re-dequantising them every token.
G=/root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
TANG_GPUS=2 TANG_GLM5_CACHE=1 target/release/tang-llm flash-generate $G \
    --prompt "Write a haiku about the sea." -n 32 --temp 0 -v --out /root/g2.ids
# (TANG_SPLIT_AT=23 is the balanced default it picks itself; set it to override)

# 5. split-vs-reference check: the same greedy ids from the CPU reference engine
TANG_GLM5_GPU=0 TANG_GLM5_CACHE=1 target/release/tang-llm flash-generate $G \
    --prompt "Write a haiku about the sea." -n 8 --temp 0 --out /root/cpu.ids
head -c 60 /root/cpu.ids; head -c 60 /root/g2.ids   # first 8 ids must match

# 6. throughput (naive; the number to beat, not a result)
TANG_GPUS=2 TANG_GLM5_CACHE=1 target/release/tang-llm flash-bench $G \
    --prompt-ids "1000 1001 1002" -n 64
```

Notes:

- `glm5-ref` takes the *first shard* and finds the rest. Metadata comes from shard 1;
  the 15 tensors of shard 6 (MTP) are found by the same discovery.
- `flash-generate` / `flash-bench` route a `glm5-next` GGUF to `glm5::engine::Glm5Engine`
  (bin dispatch, any build), not to the qwen3next `Coder` / Flash-Next `Engine`. The engine
  is resumable: KDA state + conv history per layer, DSA latent + indexer caches per layer.
  It is checked against the full reference forward on the tiny model (chunked prefill +
  decode: logits ≤3e-5, greedy identical).
- What runs where (naive-but-correct, GATES.md):
  - **GPU** (cuda build), on the layer's device (`TANG_SPLIT_AT`):
    - every FFN: the dense lead-in and all routed + shared experts (W5's `DevMoe`, GGUF
      bytes resident in VRAM);
    - KDA: conv + delta core (W4). Prefill runs in 64-token GPU chunks, decode uses the GPU
      step kernel, and a sub-64 tail runs the CPU core on the same state;
    - DSA: latent append, pooled indexer (keys, pools, selection) and attention (W3), with
      the latent cache in **bf16** (W3's format);
    - mHC pre/post (W2).
  - **CPU f32**: embedding, the projection GEMVs around each GPU kernel (KDA q/k/v/gates/
    gated norm/o_proj, MLA q_a/q_b/o_proj), RMSNorms, head.
  - Naive: every kernel stages its inputs and caches host↔GPU per call, and recurrent and
    latent state lives on the host. That's slow but correct. Next is making weights and
    caches device-resident and the layer loop host-free, then fusing (GATES.md "FAST").
- Switches: `TANG_GLM5_GPU=0` runs everything on the CPU. `TANG_GLM5_GPU_KDA=0`,
  `TANG_GLM5_GPU_MLA=0` and `TANG_GLM5_GPU_MHC=0` each move one component back to the CPU,
  for bisecting a GPU mismatch. `TANG_GLM5_CACHE=1` caches mixer weights.
- GPU-vs-CPU bars (cuda tests in `glm5::engine`): FFN + KDA + mHC are f32, so ≤1e-3 logits
  and identical top-1. With MLA on, the GPU latents are bf16, so the bar is ≥98% top-1 and
  logits ≤0.25 (both printed).
- Type-checking the cuda build without a toolkit: `CUDARC_CUDA_VERSION=12080 cargo check
  -p tang-llm --no-default-features --features cuda --lib --tests --bins`.
- Tokenizer: `tokenizer.ggml.pre = "glm4"`, supported since aff94dd6 (`tokenize.rs`); the
  chat template ships in the GGUF (`encode_chat`).
- For the engine the real quant mix is Q8_0 (attention/projections), Q4_K/Q5_K (gate/up
  experts), Q5_K/Q6_K (down experts) — all in `GgmlType::can_dequantize`; the CUDA engine
  uploads native blocks for its kernel types (Q8_0 direct; Q4_K/Q5_K/Q6_K through requant,
  same as Coder-Next's pipeline).
