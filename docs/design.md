# The compiler as a design tool: sequence mixers

Question: if we only keep layer variants that tang derives a one-pass
(flash-style) or chunked-scan kernel for — forward and backward — does that
filter find a sequence mixer that beats strong baselines at equal compute?

Short answer: **no.** Among the attention variants the filter keeps, the
best ones (forget-gated softmax or sigmoid, i.e. Forgetting-Transformer-style
gates) beat causal softmax by about 0.02 nats at 1B tokens. But every
recurrent baseline — gated delta net, Mamba-2's SSD, GLA — beats all of them
by a further 0.03–0.06 nats, with ~23% more parameters. At equal wall-clock in our
PyTorch setup, plain softmax with FlashAttention wins outright, because the
candidates ran without a flash kernel in torch. The compiler was useful as a
filter and as a diagnostic (§2). It did not find a better layer.

## 1. Search space

`crates/tang-derive-wasm/src/design.rs` enumerates two families.

**Attention family**: `O = norm(score(s, i, j, u, w), mask)·v`, with `s` the
scaled `q·k`, `i, j` positions, and `u = g[i]`, `w = g[j]` a per-position
input vector (a learned data-dependent gate; with `g` the cumulative log
forget gate, `s + u − w` is the Forgetting Transformer).

- 10 score functions: plain, softcap `30 tanh(s/30)`, ALiBi, forget `s+u−w`,
  key bias `s+w`, key scale `s·σ(w)`, forget+softcap, forget+ALiBi,
  log-distance `s − log(1+i−j)`, sigmoid bias `s − log n`;
- 3 masks: causal, causal window 128, prefix-LM 64;
- 6 normalizers: softmax, softmax1 (`exp/(1+Σexp)`), sigmoid (unnormalized),
  normalized sigmoid, `relu²`, identity.

180 variants. Each is built as the naive graph (every `n×n` intermediate
written), run through tang's normal pass pipeline (`derive_graph`, the
FlashAttention-2 memory schedule), and judged by counting `n×n` buffers in
the derived program — forward, backward through q/k/v, and backward
including the gate gradient. Nothing in the pipeline knows what any of these
layers is.

**Scan family**: `S_t = α_t(I − [δ]β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ`,
`o = Sᵀq` — all 8 on/off combinations of `α`, `β`, `δ` (linear attention,
RetNet-like, SSD/Mamba-2, DeltaNet, gated DeltaNet, …), judged by whether
`chunked_scan_plan` takes it (chunked tensor-core forward + derived chunked
backward).

Full table: [`docs/design/space.csv`](design/space.csv)
(`cargo test --release -p tang-derive-wasm design_space_table -- --nocapture`).

## 2. What the filter says

| normalizer | lifts (fwd + dq dk dv) | lifts incl. gate gradient |
|---|---|---|
| softmax | 30/30 | 15/30 |
| sigmoid | 30/30 | 15/30 |
| relu² | 30/30 | 15/30 |
| identity | 30/30 | 15/30 |
| softmax1 | 0/30 | 0/30 |
| normalized sigmoid | 0/30 | 0/30 |

All 8 scan variants chunk. 128/180 attention variants lift for q, k, v;
68/180 also lift the gate gradient. These are exactly the ungated scores
under the four lifting normalizers; no gated variant lifts it.

Three things the filter tells a designer, none of which we told it:

1. **Data-dependent gates are free in the forward and in dq/dk/dv, but not
   in their own gradient.** `∂g` is a row sum plus a column sum of `dS`; tang
   fuses the column reduction that produces `dk` but not a reduction of `dS`
   to a vector, so it materializes `n×n` (2 buffers). Hand-written FoX
   kernels compute it as an extra epilogue; tang can't yet. A rewrite that
   makes it lift: put the gate in the features (`q' = [q, √d·c_i, √d]`,
   `k' = [k, 1, −c_j]`), which turns the gate gradient into part of `dq`/`dk`
   (with a precision cost in bf16 for long sequences).
2. **softmax1 does not lift**, because `max(rowmax, 0)` breaks the
   `stabilize_softmax` pattern; equivalently it is softmax with one extra key
   of score 0 and value 0 — a sink token — which does lift. **Normalized
   sigmoid does not lift** (two independent sums over keys are not merged
   into one online pass).
3. Masks and positional score terms never matter for derivability.

## 3. Kernel speed (tang, RTX 3090)

`crates/tang-bench/examples/design_bench.rs`, causal, d = 64, one head,
fp32/TF32 tiles via `Program::tuned`. Raw: [`docs/design/kernels_3090.csv`](design/kernels_3090.csv).
The 3090 is shared with other jobs, so treat ±2× at n = 4096 as noise.

| variant | fwd ms n=1024 | fwd+bwd ms n=1024 | fwd ms n=2048 | fwd+bwd ms n=2048 |
|---|---|---|---|---|
| softmax | 0.032 | 0.142 | 0.065 | 0.270 |
| forget | 0.032 | 0.177 | 0.064 | 0.270 |
| softcap_forget | 0.043 | 0.219 | 0.093 | 1.00 |
| logdist | 0.038 | 0.187 | 0.070 | 0.490 |
| sigmoid | 0.046 | 0.146 | 0.096 | 0.277 |
| forget_sigmoid | 0.041 | 0.168 | 0.337 | 0.344 |

Every candidate is one forward kernel with no `n×n` buffer; the derived
backward is 8 kernels (softmax family) or 5 (sigmoid). At n = 1024 the gated
and sigmoid variants cost within ~25% of plain causal softmax.

## 4. Training

Setup (`scripts/design/`): GPT-style decoder, 12 layers, d = 768, 12
heads, RoPE (attention variants), RMSNorm, 4× GELU MLP, tied 50304-token
GPT-2 embedding, seq 1024, batch 64×1024 tokens, AdamW (lr 1.5e-3, wd 0.1,
β 0.9/0.95), cosine to 10%, 1B tokens of FineWeb-Edu (`sample-10BT`, first two
shards, 1.49B tokens train / 10M held out), bf16 autocast, one seed. Only the
mixer changes. The hardware was RTX PRO 6000 Blackwell (one run per GPU, RunPod secure cloud).
`prep.py` tokenizes, `train.py MIXER` trains, `run.sh` is the pod entrypoint,
`analyze.py` makes the table. Raw eval curves: `docs/design/runs/*.jsonl`.

Mixers:

- **softmax**: causal SDPA (FlashAttention). Baseline.
- **gdn**: fla `GatedDeltaNet` (6 heads × 128, expand_v 2, short conv). Baseline.
- **ssd**: Mamba-2's SSD core (scalar per-head decay) via fla
  `chunk_simple_gla`, inside gdn's projections, conv, gate and norm, so it differs
  from gdn only by the delta rule. Baseline. (fla's own `Mamba2` layer fell back
  to its PyTorch path without `mamba_ssm` kernels and ran out of memory, so we used this instead.)
- **gla**: the same wrapper with per-channel decay (fla `chunk_gla`). This is the
  diagonal-decay scan that discover (PR #181) finds a chunked lift for. tang has no
  chunked CUDA kernel for it, and its backward is a reverse scan.
- From the filter's survivors: **forget** `s + c_i − c_j` (FoX), **softcap_forget**,
  **logdist** `s − log(1+i−j)`, **sigmoid** `σ(s − log T)`, **forget_sigmoid**.
- **swvar**: softmax attention plus a learned `w ⊙ Var_p[v]` readout. This is
  discover's softmax-weighted variance lift, which has a derived one-pass backward.
  It runs as one SDPA over `[v, v²]`.

Gated and sigmoid variants ran as explicit masked attention in PyTorch
(torch.compile). torch 2.8's flex_attention cannot backprop into a captured
gate tensor. logdist ran through flex_attention.

| mixer | params | tokens | val loss (1B tok) | Δ vs softmax | tok/s | wall s | val @ softmax wall |
|---|---|---|---|---|---|---|---|
| gdn | 152.2M | 1.00B | 3.239 | -0.077 | 109k | 9329 | 3.491 |
| gla | 159.1M | 1.00B | 3.248 | -0.069 | 91k | 11097 | 3.556 |
| ssd | 152.1M | 1.00B | 3.265 | -0.051 | 102k | 9933 | 3.542 |
| forget_sigmoid | 123.7M | 1.00B | 3.294 | -0.022 | 142k | 7113 | 3.468 |
| forget | 123.7M | 1.00B | 3.297 | -0.020 | 136k | 7445 | 3.486 |
| softcap_forget | 123.7M | 1.00B | 3.297 | -0.019 | 133k | 7632 | 3.494 |
| logdist | 123.6M | 1.00B | 3.306 | -0.010 | 167k | 6069 | 3.420 |
| swvar | 123.6M | 1.00B | 3.313 | -0.004 | 192k | 5284 | 3.376 |
| softmax | 123.6M | 1.00B | 3.316 | +0.000 | 245k | 4140 | 3.316 |
| sigmoid | 123.6M | 1.00B | 3.347 | +0.030 | 163k | 6224 | 3.469 |

Losses are validation cross-entropy (nats) on the held-out 10M tokens (40×32
sequences). "val @ softmax wall" is each run's loss at the moment the softmax
run finished.

Reading it honestly:

- **Equal tokens.** All runs see the same 1B tokens. The softmax-family candidates have
  the same parameter count as softmax (+0.1M for the gate) and the same matmul
  FLOPs. forget / forget_sigmoid / softcap_forget improve on softmax by 0.019–0.022,
  logdist by 0.010 and swvar by 0.004. sigmoid is worse by 0.030. This is one seed,
  so differences under ~0.01 are not distinguishable from noise; the gate's
  ~0.02 gain matches what the Forgetting Transformer paper reports. The forget
  gate is the only clear win the filter surfaced, and it is a known layer.
- **The recurrent baselines win.** gdn 3.239, gla 3.248 and ssd 3.265 all beat every
  attention variant. They have about 23% more parameters (fla's default
  expand_v = 2 projections), so this is not an equal-FLOPs comparison. But even
  the gap from ssd to the best attention variant (0.03) is larger than the
  gain any attention variant made over softmax.
- **Equal wall-clock.** softmax runs at 245k tok/s with FlashAttention. The gated and
  sigmoid variants run at 133–167k tok/s because torch has no fused kernel for
  them. By the time softmax finishes, every other run has a higher loss. This penalizes
  the implementation, not the layer: tang's derived kernels for these variants
  cost 1.0–1.25× causal softmax at n = 1024 for fwd+bwd (§3). With such
  kernels the wall-clock picture would track the equal-token one, minus up to ~25% of
  attention time. We did not run training through tang's kernels (no PyTorch
  binding for the derived programs). That is the main gap in this study.
- **The answer to the question.** The derivability filter is
  cheap (188 variants judged in 0.1 s on a laptop) and it correctly separates what
  gets a one-pass kernel from what doesn't, with useful diagnostics: the gate
  gradient, softmax1 and normalized sigmoid. But "liftable" is a weak prior on
  "good". Most of the 128 survivors are reparameterizations of softmax attention
  that train about the same, and the variants that were better at this scale (delta-rule
  and decay recurrences) were already known and already had hand-written
  kernels. The compiler did not find a sequence mixer that beats gated delta net
  or Mamba-2 at equal compute. That is a negative result.

Spend: about $115 of RunPod (8× + 3× + 1× + 2× RTX PRO 6000, 3–3.4 h each, two
stuck pods replaced). All pods are terminated.

## 5. Follow-ups

- A PyTorch autograd binding for tang's derived attention programs, so the
  candidates train at kernel speed and the equal-wall-clock column is
  meaningful.
- Lift the gate gradient: fuse a row/column sum of `dS` into the dq/dk tiles,
  or use the feature-augmentation rewrite from §2.
- Lift softmax1 by recognizing `max(m, 0)` as one extra key, and lift normalized
  sigmoid by merging same-axis sums into one online pass.
- Search the space discover opens (PR #181): attention × scan compositions, and
  diag+delta (KDA-style) chunking, which it does not cover yet.
- Multiple seeds and a larger scale (350M+, 10B+ tokens) before trusting any
  difference under ~0.02.
