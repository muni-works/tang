# GLM-5.3-Flash (glm5-next) GGUF map for tang

W0 harness deliverable. Every GGUF metadata key and tensor for `general.architecture = glm5-next`,
mapped to the Hugging Face parameter and to what tang needs. Sources: llama.cpp `src/llama-arch.cpp`
/ `src/models/glm5-next.cpp` / `conversion/glm.py` + `gguf-py` (read-only shared checkout), the real
`unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL **shard headers** (fetched via HTTP Range requests; no
weights downloaded), and `zai-org/GLM-5.3-Flash` config.json / tokenizer + safetensors headers.

The tiny fixture model that mirrors all of this lives in `/private/tmp/glm5-tiny/`
(built by `crates/tang-llm/scripts/glm5next_tiny.py`).

## 1. Metadata keys (GGUF KV -> meaning -> value on the real model)

All keys are prefixed `glm5-next.` except `general.*` and `split.*`.

| GGUF key | Type | Real value (UD-Q4_K_XL shard 1) | HF config source | tang need |
|---|---|---|---|---|
| `general.architecture` | str | `glm5-next` | `architectures[0]=Glm5NextForConditionalGeneration` | Arch enum |
| `general.file_type` | u32 | 15 (mostly Q8_0) | quantize output type | mixed per-tensor types; parse per-tensor |
| `glm5-next.block_count` | u32 | 46 | `num_hidden_layers=45` **+ `num_nextn_predict_layers=1`** | 45 trunk + 1 MTP |
| `glm5-next.embedding_length` | u32 | 4096 | `hidden_size` | hidden H |
| `glm5-next.feed_forward_length` | u32 | 12288 | `intermediate_size` (dense FFN) | dense lead layers |
| `glm5-next.context_length` | u32 | 1048576 | `max_position_embeddings` | ctx |
| `glm5-next.vocab_size` | u32 | 154880 | `vocab_size` | vocab V |
| `glm5-next.attention.head_count` | u32 | 64 | `num_attention_heads` | heads (KDA and DSA) |
| `glm5-next.attention.head_count_kv` | u32 array | `[0,0,0,1,0,0,0,1,...,1]` (46) | derived: 0 for `layer_types==linear_attention`, 1 for DSA, padded 1 for MTP | **layer type flag**: `n_head_kv(i)==0` => KDA (recurrent), else DSA |
| `glm5-next.attention.layer_norm_rms_epsilon` | f32 | 1e-5 | `rms_norm_eps` | RMSNorm eps |
| `glm5-next.attention.layer_norm_epsilon` | f32 | 1e-6 | hard-coded in converter (LayerNorm eps of indexer `k_norm`) | indexer LayerNorm eps |
| `glm5-next.attention.q_lora_rank` | u32 | 1536 | `q_lora_rank` | DSA q lora |
| `glm5-next.attention.kv_lora_rank` | u32 | 512 | `kv_lora_rank` | DSA kv latent (NoPE; the KV cache stores this) |
| `glm5-next.attention.rope.dimension_count` | u32 | 0 | `qk_rope_head_dim=0` (NoPE) | **no rope at all** |
| `glm5-next.attention.key_length` | u32 | 512 | `kv_lora_rank + qk_rope_head_dim` | (redundant) |
| `glm5-next.attention.value_length` | u32 | 512 | `kv_lora_rank` | (redundant) |
| `glm5-next.attention.key_length_mla` | u32 | 256 | `qk_nope_head_dim + qk_rope_head_dim` | per-head K dim after decompress |
| `glm5-next.attention.value_length_mla` | u32 | 256 | `v_head_dim` | per-head V dim |
| `glm5-next.attention.indexer.head_count` | u32 | 32 | `index_n_heads` | indexer heads |
| `glm5-next.attention.indexer.key_length` | u32 | 128 | `index_head_dim` | indexer head dim |
| `glm5-next.attention.indexer.top_k` | u32 | 2048 | `index_topk` | selected tokens per query |
| `glm5-next.attention.indexer.kpool` | u32 | 4 | `index_kpool` | pool compression width |
| `glm5-next.attention.indexer.kpool_select_tail` | bool | true | `index_kpool_always_select_tail` | append incomplete tail pool |
| `glm5-next.attention.indexer.types` | bool array | all true (46) | `indexer_types` ("full"/"shared") | full = run indexer; shared = reuse previous full layer's top-k |
| `glm5-next.attention.indexer.index_share_mtp` (unsloth) | u32 | 1 | `index_share_for_mtp_iteration` | (unsloth-added, not in llama.cpp trunk converter) |
| `glm5-next.hyper_connection.count` | u32 | 4 | `hc_mult` | mHC streams N |
| `glm5-next.hyper_connection.sinkhorn_iterations` | u32 | 20 | `hc_sinkhorn_iters` | Sinkhorn iters |
| `glm5-next.hyper_connection.epsilon` | f32 | 1e-6 | `hc_eps` | hc eps |
| `glm5-next.kda.head_dim` | u32 | 128 | `linear_attn_config.head_dim` | KDA per-head dim |
| `glm5-next.kda.gate_lower_bound` | f32 | -5.0 | `linear_attn_config.gate_lower_bound` | safe gate lb (absent => softplus path) |
| `glm5-next.ssm.conv_kernel` | u32 | 4 | `linear_attn_config.short_conv_kernel_size` | short conv k |
| `glm5-next.expert_count` | u32 | 288 | `n_routed_experts` | routed experts |
| `glm5-next.expert_used_count` | u32 | 8 | `num_experts_per_tok` | top-k |
| `glm5-next.expert_shared_count` | u32 | 1 | `n_shared_experts` | shared expert (its FFN width = moe FFN) |
| `glm5-next.expert_feed_forward_length` | u32 | 2048 | `moe_intermediate_size` | expert FFN |
| `glm5-next.expert_group_count` | u32 | 1 | `n_group` | (1 = no grouping) |
| `glm5-next.expert_group_used_count` | u32 | 1 | `topk_group` | (1 = no grouping) |
| `glm5-next.expert_weights_scale` | f32 | 2.5 | `routed_scaling_factor` | router scale |
| `glm5-next.expert_weights_norm` | bool | true | `norm_topk_prob` | normalize top-k weights |
| `glm5-next.expert_gating_func` | u32 | 2 (sigmoid) | `scoring_func="sigmoid"` (converter: absent => llama.cpp defaults sigmoid) | router gating |
| `glm5-next.leading_dense_block_count` | u32 | 3 | `first_k_dense_replace` | first 3 layers dense FFN |
| `glm5-next.nextn_predict_layers` | u32 | 1 | `num_nextn_predict_layers` | MTP layer count (blocks 45..45) |
| `glm5-next.swiglu_clamp_exp` | f32 array(46) | all 10.0 | `swiglu_limit` | clamp for dense+expert gate/up |
| `glm5-next.swiglu_clamp_shexp` | f32 array(46) | all 10.0 | `swiglu_limit` | clamp for shared expert |
| `quantize.imatrix.*` | — | present | — | ignore |
| `split.count` / `split.no` / `split.tensors.count` | u32 | 6 / 0..5 / 1412 | — | shard reassembly (tang mmap must handle) |

Note: `head_count_kv` **doubles as the KDA/DSA per-layer flag** in llama.cpp
(`hparams.is_recr(i) = n_head_kv(i) == 0`). The array is padded with 1s to
`block_count` for the MTP block (MTP is a DSA layer).

## 2. Tensor map (GGUF name -> shape/type -> HF parameter)

GGUF dims below are **ne order** (ne0 = rows = innermost); the converter writes HF
`nn.Linear.weight [out, in]` as ne `[in, out]`. `N` = block index (0..44 trunk, 45 = MTP).
Quant types are the **real per-tensor types from the UD-Q4_K_XL shard headers**.

### Global

| GGUF tensor | ne dims (real) | type (UD) | HF parameter |
|---|---|---|---|
| `token_embd.weight` | [4096, 154880] | Q8_0 | `model.language_model.embed_tokens.weight` |
| `output_norm.weight` | [4096] | F32 | `model.language_model.norm.weight` (RMSNorm) |
| `output.weight` | [4096, 154880] | Q8_0 | `lm_head.weight` (untied) |

### Per-block, KDA (linear attention) layers — 34 of them (N % 4 != 3, plus layer 44)

`head_dim`=128, `d_inner = head_dim * 64 heads = 8192`, `k=4`, `A_log` per head (64),
`dt_bias` per channel (8192).

| GGUF tensor | ne dims (real) | type (UD) | HF parameter | notes for tang |
|---|---|---|---|---|
| `blk.N.attn_norm.weight` | [4096] | F32 | `layers.N.input_layernorm.weight` | RMSNorm (weighted) |
| `blk.N.attn_q.weight` | [8192, 4096] | Q8_0 | `self_attn.q_proj.weight` | |
| `blk.N.attn_k.weight` | [8192, 4096] | Q8_0 | `self_attn.k_proj.weight` | |
| `blk.N.attn_v.weight` | [8192, 4096] | Q8_0 | `self_attn.v_proj.weight` | |
| `blk.N.attn_output.weight` | [4096, 8192] | Q8_0 | `self_attn.o_proj.weight` | |
| `blk.N.ssm_conv1d_q.weight` | [4, 1, 8192, 1] | **F32** | `self_attn.q_conv1d` (synthesized by converter from `conv1d.weight` [768·3? no: qkv conv]) | per-channel depthwise conv k=4; stored 4D, reshape 2D at runtime |
| `blk.N.ssm_conv1d_k.weight` | [4, 1, 8192, 1] | F32 | idem k | |
| `blk.N.ssm_conv1d_v.weight` | [4, 1, 8192, 1] | F32 | idem v | |
| `blk.N.ssm_f_a.weight` | [128, 4096] | Q8_0 | `self_attn.forget_gate.f_a_proj.weight` (4096->128) | forget-gate low-rank a |
| `blk.N.ssm_f_b.weight` | [8192, 128] | Q8_0 | `self_attn.forget_gate.f_b_proj.weight` (128->8192) | forget-gate low-rank b |
| `blk.N.ssm_a` | [64] | **F32** | `self_attn.forget_gate.A_log` transformed: stored as `-exp(A_log)` | safe-gate path (W4): llama.cpp computes `lb * sigmoid(-ssm_a * g)` == HF's `lb * sigmoid(exp(A_log)*g)`; with A_log=0 the tensor is -1.0 per head |
| `blk.N.ssm_dt.bias` | [8192] | **F32** | `self_attn.forget_gate.dt_bias` | kept exact |
| `blk.N.ssm_beta.weight` | [64, 4096] | Q8_0 | `self_attn.b_proj.weight` (4096->64, sigmoid => beta) | |
| `blk.N.ssm_g_a.weight` | [128, 4096] | Q8_0 | `self_attn.g_a_proj.weight` | output gate low-rank a |
| `blk.N.ssm_g_b.weight` | [8192, 128] | Q8_0 | `self_attn.g_b_proj.weight` | output gate low-rank b |
| `blk.N.ssm_norm.weight` | [128] | F32 | `self_attn.o_norm.weight` | RMSNormGated (sigmoid gate) per head |

Converter note (W4): llama.cpp's `conversion/glm.py` synthesizes `q_conv1d/k_conv1d/v_conv1d`
from HF's **single fused** `self_attn.conv1d.weight` `[3*d_inner, 1, k]` (the HF module has ONE
conv over the concatenated qkv). Tang's loader must replicate that split (q | k | v channels).
Beware the published zai checkpoints also carry a fused `qkv_proj`/`fused_qkvbfg_a_proj`
(the `modules_to_not_convert` list shows `qkv_proj` and `fused_qkvbfg_a_proj`) — the converter
handles the plain `q_proj/k_proj/v_proj/...` layout; tang should read the same.

### Per-block, DSA (MLA-NoPE + indexer) layers — 11 of them (N % 4 == 3, N in 3,7,11,...,43) + MTP block 45

`q_lora 1536, kv_lora 512 (NoPE), qk_nope 256, v_head 256, 64 heads; indexer 32 heads x 128,
kpool 4, top-2048`.

| GGUF tensor | ne dims (real) | type (UD) | HF parameter | notes |
|---|---|---|---|---|
| `blk.N.attn_norm.weight` | [4096] | F32 | `layers.N.input_layernorm.weight` | |
| `blk.N.attn_q_a.weight` | [1536, 4096] | Q8_0 | `self_attn.q_a_proj.weight` | 4096->1536 |
| `blk.N.attn_q_a_norm.weight` | [1536] | F32 | `self_attn.q_a_layernorm.weight` | RMSNorm |
| `blk.N.attn_q_b.weight` | [16384, 1536] | Q8_0 | `self_attn.q_b_proj.weight` | 1536 -> 64*(256+0) |
| `blk.N.attn_kv_a_mqa.weight` | [512, 4096] | Q8_0 | `self_attn.kv_a_proj_with_mqa.weight` | 4096 -> 512 (+0 rope, NoPE) |
| `blk.N.attn_kv_a_norm.weight` | [512] | F32 | `self_attn.kv_a_layernorm.weight` | RMSNorm over latent |
| `blk.N.attn_k_b.weight` | [256, 512, 64] | Q8_0 | split of `self_attn.kv_b_proj.weight` [64*(256+256), 512] | converter splits kv_b into per-head k_b (transposed to [256, 512] per head) and v_b |
| `blk.N.attn_v_b.weight` | [512, 256, 64] | Q8_0 | split of `kv_b_proj` (value half) | |
| `blk.N.attn_output.weight` | [4096, 16384] | Q8_0 | `self_attn.o_proj.weight` | 64*256 -> 4096 |
| `blk.N.indexer.attn_k.weight` | [128, 4096] | Q8_0 | `self_attn.indexer.wk.weight` | 4096 -> 128 (single "head") |
| `blk.N.indexer.attn_q_b.weight` | [4096, 1536] | Q8_0 | `self_attn.indexer.wq_b.weight` | 1536 -> 32*128 |
| `blk.N.indexer.k_norm.weight` | [128] | F32 | `self_attn.indexer.k_norm.weight` | **LayerNorm** (not RMS) |
| `blk.N.indexer.k_norm.bias` | [128] | F32 | `self_attn.indexer.k_norm.bias` | LayerNorm bias |
| `blk.N.indexer.proj.weight` | [32, 4096] | **F32** | `self_attn.indexer.weights_proj.weight` | head-weight mixing |
| `blk.N.indexer_compressor_gate.weight` | [128, 4096] | Q8_0 | `self_attn.indexer.index_kpool_compress_gate` | gate scores per token |
| `blk.N.indexer_compressor_ape.weight` | [128, 4] | F32 | `self_attn.indexer.index_kpool_compress_ape` | [head_dim, kpool] position bias |

KV cache (W3): stores the **512-dim latent** (+ no rope part); decompress per head via
`k_b/v_b` (64 heads each 256/256). NoPE means **no positional encoding anywhere in DSA**.

### Per-block, mHC (every trunk block 0..44; **not** the MTP block)

HF-name note: the zai safetensors store these **flat** (`model.layers.N.hc_attn_fn`),
matching the GGUF side 1:1; transformers 5.2x re-nests them as `attn_hc.fn` at load time.

`hc = 4`, `hc_mix_dim = (2+hc)*hc = 24`.

| GGUF tensor | ne dims (real) | type (UD) | HF parameter | notes |
|---|---|---|---|---|
| `blk.N.hc_attn_fn.weight` | [16384, 24] | Q8_0 | `layers.N.hc_attn_fn` [24, 4*4096] (HF Linear weight [out, in]) | ne = [in=hc*H, out=24] |
| `blk.N.hc_attn_base.weight` | [24] | F32 | `layers.N.hc_attn_base` | bias [24] |
| `blk.N.hc_attn_scale.weight` | [3] | F32 | `layers.N.hc_attn_scale` | (pre, post, comb) scales |
| `blk.N.hc_ffn_fn.weight` | [16384, 24] | Q8_0 | `layers.N.hc_ffn_fn` [24, 4*4096] | |
| `blk.N.hc_ffn_base.weight` | [24] | F32 | `layers.N.hc_ffn_base` | |
| `blk.N.hc_ffn_scale.weight` | [3] | F32 | `layers.N.hc_ffn_scale` | |

HF sidecar: the **input norm** of the HC module (unweighted RMSNorm, eps=rms_norm_eps) has **no
tensor** — it is weight-free. The final stream collapse (`hc_head`) is an **unweighted mean
over the 4 streams** (no tensors either). W2 owns the exact math.

### Per-block, FFN — dense lead (N in 0..2) and MTP-adjacent trunk

| GGUF tensor | ne dims (real) | type (UD) | HF parameter |
|---|---|---|---|
| `blk.N.ffn_norm.weight` | [4096] | F32 | `layers.N.post_attention_layernorm.weight` |
| `blk.N.ffn_gate.weight` | [12288, 4096] | Q8_0 | `mlp.gate_proj.weight` |
| `blk.N.ffn_up.weight` | [12288, 4096] | Q8_0 | `mlp.up_proj.weight` |
| `blk.N.ffn_down.weight` | [4096, 12288] | Q8_0 | `mlp.down_proj.weight` |

Dense FFN uses the same **swiglu clamp (limit 10)** semantics: gate clamped to
max 10, up clamped to [-10, 10], then `silu(gate) * up`.

### Per-block, FFN — sparse MoE (N in 3..44 + MTP 45)

| GGUF tensor | ne dims (real) | type (UD) | HF parameter | notes |
|---|---|---|---|---|
| `blk.N.ffn_gate_inp.weight` | [288, 4096] | **F32** | `mlp.gate.weight` | router |
| `blk.N.exp_probs_b.bias` | [288] | F32 | `mlp.gate.e_score_correction_bias` | noaux_tc bias |
| `blk.N.ffn_gate_exps.weight` | [4096, 2048, 288] | **Q4_K** (40 layers: 3–10, 13–43, 45) / Q5_K (layer 11) | `mlp.experts.gate_proj` stacked [288, 2048, 4096] | gate per expert |
| `blk.N.ffn_up_exps.weight` | [4096, 2048, 288] | Q4_K (40) / Q5_K (11) | `mlp.experts.up_proj` stacked | up per expert |
| `blk.N.ffn_down_exps.weight` | [2048, 4096, 288] | **Q5_K** (40 layers) / Q6_K (layers 11, 12, 44) | `mlp.experts.down_proj` stacked [288, 4096, 2048] | down per expert |
| `blk.N.ffn_gate_shexp.weight` | [2048, 4096] | Q8_0 | `mlp.shared_experts.gate_proj.weight` | shared expert |
| `blk.N.ffn_up_shexp.weight` | [2048, 4096] | Q8_0 | `mlp.shared_experts.up_proj.weight` | |
| `blk.N.ffn_down_shexp.weight` | [4096, 2048] | Q8_0 | `mlp.shared_experts.down_proj.weight` | |

**UD-Q4_K_XL per-tensor quant summary** (from shard headers, all 1412 tensors):

- F32 exact (never quantized): all norms (`attn_norm`, `ffn_norm`, `attn_q_a_norm`,
  `attn_kv_a_norm`, `output_norm`), `ssm_a`, `ssm_dt.bias`, `ssm_conv1d_{q,k,v}`,
  `ssm_norm`, `exp_probs_b.bias`, `ffn_gate_inp`, `indexer.k_norm.{weight,bias}`,
  `indexer.proj`, `indexer_compressor_ape`, `hc_attn/ffn_{base,scale}`, `nextn.*` norms.
- Q8_0: everything else except routed experts — embeddings + lm_head (674 MB each),
  KDA q/k/v/o, f/g/beta projections, DSA q_a/q_b/kv_a/o, k_b/v_b (12 layers),
  hc_attn_fn/hc_ffn_fn, indexer.attn_k/attn_q_b/compressor_gate, dense ffn (3 layers),
  shared experts, MTP block weights.
- Routed experts: gate/up = **Q4_K** everywhere except layer 11 (Q5_K); down = **Q5_K**
  everywhere except layers 11, 12, 44 (Q6_K).
- No IQ_* / Q6_K-elsewhere in this file. (Other UD quants in the repo differ; this map
  is only for UD-Q4_K_XL.)
- Shard layout: 6 shards; **shard 1 (9.4 MB) holds all 73 metadata KVs and 0 tensors**;
  shards 2..6 carry 397/338/321/342/15 tensors respectively (1412 total).

**Per-layer byte budget at UD-Q4_K_XL (for W6's split plan), measured from the shard
tensor headers** (element counts × per-quant bytes; sums to ~199.7 GB / ~186 GiB of
tensor payload + alignment = the ~244 GB file):

| family | bytes |
|---|---|
| routed experts (gate/up Q4_K, down Q5_K) | **4.379 GB per sparse layer** × 43 layers = 189.9 GB |
| dense FFN layers 0–2 | 0.160 GB per layer (3×) |
| KDA attention core (q/k/v/o) | 0.143 GB per KDA layer (34×) |
| DSA attention core (q_a/q_b/kv_a/k_b/v_b/o) | 0.125 GB per DSA layer (11×) + MTP |
| KDA ssm extras (f/g/beta/conv/dt/A) | 0.004 GB per KDA layer |
| DSA indexer (all 7 tensors) | 0.0083 GB per DSA layer |
| shared expert (gate/up/down Q8_0) | 0.027 GB per sparse layer |
| mHC (2×fn Q8_0 + base/scale F32) | 0.0008 GB per trunk layer |
| norms + router + bias (F32) | ~0.005 GB per layer |
| `token_embd` + `output` (Q8_0 each) | 0.674 GB each |
| whole MTP block 45 (DSA + MoE + nextn) | 4.579 GB |

Full per-quant totals: Q4_K 114.15 GB, Q5_K 69.76 GB, Q8_0 9.62 GB, Q6_K 5.95 GB
(layers 11/12/44 `ffn_down_exps`), F32 0.23 GB. A typical whole layer: KDA+MoE
(blk.4) 4.56 GB, DSA+MoE (blk.3) 4.54 GB, KDA+dense (blk.0) 0.31 GB.

## 3. NextN / MTP block (45)

| GGUF tensor | ne dims (real) | type (UD) | source |
|---|---|---|---|
| `blk.45.nextn.eh_proj.weight` | [8192, 4096] | Q8_0 | HF `layers.45.eh_proj` (Linear 2*H -> H) |
| `blk.45.nextn.enorm.weight` | [4096] | F32 | embed norm |
| `blk.45.nextn.hnorm.weight` | [4096] | F32 | hidden norm |
| `blk.45.nextn.shared_head_norm.weight` | [4096] | F32 | shared head norm |
| (optional, absent in UD) | | | `nextn.embed_tokens`, `nextn.shared_head.head` |

The MTP block is a **DSA layer** (attn_q_a/q_b/kv_a/k_b/v_b/o, indexer.*), **sparse MoE**
(ffn_*_exps + shexp + gate_inp + exp_probs_b), with `attn_norm`/`ffn_norm` — and **NO hc_*
tensors** (llama.cpp never loads them; the UD GGUF carries none). KDA layers do not appear
in the MTP block. The zai HF checkpoint itself has **no `layers.45.*` tensors at all** (the
45 trunk layers only; transformers ignores them via `_keys_to_ignore_on_load_unexpected =
[r"layers\.45\.", ...]`) — the MTP weights in the unsloth GGUF come from unsloth's own
nextn source. **Our tiny fixture synthesizes block 5 the same way** (copy of the DSA layer
minus hc_*, plus fresh eh_proj [H, 2H] -> GGUF ne [2H, H]).

## 4. Divergences / gotchas for the tang loader (W1)

1. **`head_count_kv` array is the layer-type flag** (0 = KDA). Don't read it as GQA.
2. **NoPE everywhere in DSA**: `qk_rope_head_dim = 0`, `rope.dimension_count = 0`. There is
   **no RoPE in the whole model** (KDA has none either). tang's rope plumbing can be skipped.
3. **kv_b split**: the GGUF carries per-head `attn_k_b [256, 512, 64]` and `attn_v_b
   [512, 256, 64]` (k_b transposed), NOT the fused `kv_b_proj [64*(256+256), 512]`.
   k_b per head is stored **transposed** ([qk_nope, kv_lora] vs HF's [kv_lora, qk_nope] per
   head slice).
4. **KDA conv**: one HF `conv1d.weight [3*d_inner, 1, 4]` splits into 3 GGUF tensors
   (`ssm_conv1d_{q,k,v}` stored 4D `[4, 1, d_inner, 1]` = F32 exact).
5. **`ssm_a`** is stored as `-exp(A_log)` truncated to `num_attention_heads` (64) — with the
   safe gate the checkpoint value is A_log=0 => tensor = -1.0 per head. HF's runtime forget
   gate uses `exp(A_log)` as `decay_rate`; tang's KDA (W4) must match whichever convention
   survives the round-trip. **W4 must verify the exact semantics against the tiny fixture**
   (`/private/tmp/glm5-tiny`), not this doc alone.
6. **MTP block has no hc_***; loading them anyway is a tensor-count error in llama.cpp.
7. **Quantization is mixed per tensor** (Q4_K/Q5_K/Q6_K/Q8_0/F32); tang's loader must read
   per-tensor types (it already does for other K-quant models via `flash_native.rs`).
8. **Tiny-fidelity note**: the tiny GGUF is all F32, so parity vs HF is exact-path; Q8_0 and
   Q4_K_M copies exist at `/private/tmp/glm5-tiny/glm5-tiny-{q8_0,q4_k_m}.gguf` for
   smoke-testing tang's quant decoders (llama.cpp F32 greedy == HF 16/16; the quantized
   copies diverge early on this random model and are decoder smoke tests only).

## 5. Fixture paths (what W1–W5 should use)

- Tiny HF checkpoint: `/private/tmp/glm5-tiny/hf` (config.json patched with the real
  zai-only keys; 5 trunk layers + synthesized MTP block 5 in model.safetensors).
- Reference dumps: `/private/tmp/glm5-tiny/ref/` —
  `embed.bin [T,256]`, `layerB{i}.bin [T,4,256]` (mHC streams after block i),
  `final.bin [T,256]`, `logits.bin [T,154880]` (seq1 = 52 tokens), `*_seq2.bin` for the
  2052-token sequence, `meta.json` (ids, shapes). All little-endian f32.
- GGUFs: `/private/tmp/glm5-tiny/glm5-tiny-f32.gguf` (177 tensors, 46 blocks, matches the
  real tensor set minus quant), `-q8_0`, `-q4_k_m`.
- Compare tools: `crates/tang-llm/scripts/glm5_compare.py` (per-layer dumps, TANG_DUMP_DIR
  format below) and `crates/tang-llm/scripts/glm5_compare_gen.py` (R2b greedy-parity suite:
  per-prompt PASS/FAIL + first-divergence index; `--print-runner` emits the tang-llm
  flash-generate loop).
- R2b prompt suite: `tests/fixtures/glm5/prompts/` — 25 prompts (5 lengths x 5 kinds:
  code/prose/mixed/unicode/tool-JSON), HF fp32 greedy expected ids in `prompts.json` and
  per-prompt `<name>.ids` / `<name>.expected`. Suite decode is deterministic (asserted
  twice per prompt) and indexer tie-free (asserted; long prompts use a variant marker).
- llama.cpp CPU build (for extra sanity runs): `/private/tmp/glm5-harness-llama/build-cpu/bin/`
  (llama-cli, llama-quantize, llama-perplexity, llama-server).

### Indexer tie-freedom (R2 fix)

The fixture's DSA indexer is built so top-k selection has **no exact ties** between valid
pools: `indexer.wq_b` query heads are paired `(h, -h)` (keys are one shared head, so one of
each pair always has `q·k > 0` and no pool scores exactly 0.0 after relu). The generator
asserts, for every query row with more than `index_topk/kpool = 4` visible pools, that the
4th-best pool score beats the 5th by >= 1e-4 (masked pools at `finfo.min` are excluded:
rows with <= 4 visible pools select every visible pool regardless of tie order).
Seed 0: seq1 min margin 2.5e-2 (33 rows), seq2 min margin 1.50e-4 (relative 3.3e-4) at
query pos 456 over 2033 rows, 0 exact-0.0 valid scores. If tang's indexer pool scores at
seq2 pos 456 drift by more than ~1e-4 from HF, a selection flip there is a real precision
issue, not a fixture tie.

## 6. TANG_DUMP_DIR format (W1 writes, glm5_compare.py reads)

tang sets `TANG_DUMP_DIR=<dir>` and writes one raw little-endian f32 `.bin` per dump plus a
`index.json`:

```json
{
  "model": "glm5-tiny-f32.gguf",
  "token_ids": [ ... the exact input ids ... ],
  "seqs": {
    "seq1": {"len": 52, "dumps": ["embed", "layerB0", ..., "layerB4", "final", "logits"]},
    "seq2": {"len": 2052, "dumps": ["layerB0", ..., "layerB4", "final"]}
  }
}
```

- `embed.bin` `[T, H]` — embedding lookup output (stream 0; the other 3 streams are copies).
- `layerB{i}.bin` `[T, 4, H]` — the 4 mHC streams **after the whole block i** (attn + ffn),
  matching `ref/layerB{i}.bin` byte-for-byte.
- `final.bin` `[T, H]` — after `hc_head` mean + final norm.
- `logits.bin` `[T, V]`.
- Optional component dumps for the component workers' own parity gates:
  `layer{i}_attn.bin` `[T, 4, H]` (streams after attention sub-block, before ffn_hc),
  `layer{i}_mlp.bin` `[T, H]` (collapsed mlp output), etc. — anything present in both
  `TANG_DUMP_DIR` and `ref/` with the same name is compared.

The compare tool reports per-dump max abs err, max rel err, and (for `logits`) top-1
agreement vs `/private/tmp/glm5-tiny/ref`.
