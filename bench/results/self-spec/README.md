# Self-speculation drafters (CPU, mew)

Drafting with the model itself (early exit, layer skip, a harder-quantized
copy), verifying with the full model under greedy. Zero training.
Tool: `crates/tang-bench/examples/self_spec.rs` (plain f32 CPU Qwen3 forward
over the MLX 4-bit checkpoints, same approach as `jacobi_cpu.rs`).

## Verdict

**Doesn't pay, and can't stack meaningfully with prompt lookup.** Best config
is `q3-all` (every layer requantized 4 -> 3 bit, shared KV). On Qwen3-8B it
gets 96% first-draft acceptance and 5.3 accepted tokens per k=8 window, but
the drafter reads 0.80 of the full model's bytes per drafted token, so the
best tokens-per-full-read is **1.09 (k=1-2)**. Prompt lookup is 1.66 (k=4) /
1.94 (k=16). With drafter cost c = 0.8 the ceiling even at 100% acceptance is
(k+1)/(1+0.8k) < 1.25. Layer-skip / early-exit drafters are cheaper only
when they drop many layers, and then acceptance collapses (exit at half
depth: 6%).

## Metric

`tok/read = (accepted + 1) / (1 + k*c)`: tokens committed per window divided
by weight reads, counting each drafted token as `c` full-model reads (c =
drafter weight bytes / full weight bytes, lm_head included). This is the
bandwidth-bound decode model; 1.0 = plain greedy.

## Results

Qwen3-1.7B: 10 code prompts x 128 tokens, window start every 2nd position
(640 windows). Qwen3-8B: 2 prompts x 48 tokens (48 windows) for the top
three 1.7B configs. Full table in `results.tsv` (k = 1, 2, 4, 8).

| model | drafter | cost c | P(first) | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|---|
| 8B | q3-all | 0.796 | 0.96 | **1.09** | 1.08 | 1.03 | 0.86 |
| 8B | skip-every4th | 0.771 | 0.90 | 1.07 | 1.01 | 0.90 | 0.69 |
| 8B | exit-L33 | 0.923 | 0.65 | 0.86 | 0.72 | 0.55 | 0.38 |
| 1.7B | q3-all | 0.818 | 0.85 | 1.02 | 0.96 | 0.82 | 0.61 |
| 1.7B | skip-every4th | 0.795 | 0.71 | 0.95 | 0.85 | 0.67 | 0.45 |
| 1.7B | exit-L25 | 0.912 | 0.73 | 0.91 | 0.81 | 0.64 | 0.45 |
| 1.7B | q2-all | 0.636 | 0.47 | 0.90 | 0.73 | 0.50 | 0.29 |
| 1.7B | skip-odd-mid | 0.708 | 0.51 | 0.88 | 0.72 | 0.49 | 0.29 |
| 1.7B | skip-odd | 0.590 | 0.29 | 0.81 | 0.62 | 0.41 | 0.24 |
| 1.7B | exit-L14 | 0.590 | 0.06 | 0.66 | 0.49 | 0.31 | 0.18 |
| reference | prompt lookup (8B, jacobi branch) | ~0 | | | | 1.66 | 1.94 @k=16 |

`own` KV variants (drafter writes its own cache instead of reading the full
model's) are strictly worse than `shared` (skip-every4th: 0.84 vs 0.95 at k=1).

## Stacking with prompt lookup

Not measured directly (the JSON has match lengths, not tokens). Inferred:
lookup drafts cost ~0, so a stacked scheme only uses the self-drafter on
windows where lookup has no candidate; there it earns at most 1.09 tok/read
vs 1.0 for plain decode, i.e. <= 9% on those windows only, before the extra
complexity of a second resident weight copy (q3) or a gating policy.

## Verified vs inferred

- Verified: `full-sanity` drafter (all layers, 4-bit) matches greedy 8/8 on
  every window, so the drafter forward reproduces the verifier; with greedy
  verification the speculative output equals greedy by construction.
- Verified: acceptance / match-length distributions (all `*.jsonl`).
- Inferred: speedup = bytes model above (no wall-clock speculative loop;
  CPU timing on shared mew isn't representative), and the stacking bound.
- 8B sample is small (2 prompts, 48 windows); the conclusion rests on the
  cost ceiling, which doesn't depend on sample size.

## Reproduce

```
QWEN_DIR=~/.cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots/* \
RAYON_NUM_THREADS=12 STRIDE=2 nice -n 10 \
  target/release/examples/self_spec 128 10 > a.jsonl 2> a.log
DRAFTERS=full-sanity,q3-all,exit-L33,skip-every4th ... (8B) self_spec 48 2
python3 analyze.py a.jsonl b.jsonl; python3 tsv.py <model> <jsonl...>
```
