# Jacobi decoding: tokens per weight read (CPU, real weights)

**Verdict: plain Jacobi is not worth it.** On Qwen3-8B it finalizes 1.04 tokens per window at any k from 4 to 16, so 93% of the rows at k=16 are wasted. Adding an n-gram pool gets to 1.4–1.6 overall (1.6–1.9 on code, 1.1–1.2 on prose), but it costs 5–10 rows per token. Plain prompt lookup gets about the same tokens per weight read on code (1.66 at k=4, 1.94 at k=16) for 1.6–3.5 rows per token, with no forward work beyond verification. Prose gains nothing from any of these methods.

## Method
- `crates/tang-bench/examples/jacobi_cpu.rs` is a plain f32 CPU forward over the MLX 4-bit checkpoint (4-bit weights dequantized per group, every row computed independently). Each row is computed with the same arithmetic in the same order for any window size, so a row on a correct prefix reproduces the greedy logits bit for bit. Every method's output is checked token-for-token against plain greedy decoding: all runs matched (`exact` column).
- Every window costs one weight read, so tokens per iteration = tokens per weight read.
- `jacobi k`: the root (the last finalized token) plus k-1 guesses. The guesses are replaced by the model's outputs each iteration, and the prefix that agreed is accepted.
- `jacobi+pool k n G`: the k-1 guess rows are split into G branches. Branch 0 is the Jacobi carry; the other G-1 branches are continuations from an n-gram pool (n-grams of the prompt, the finalized text and every trajectory row's input→output), chained to fill the branch. Each branch has its own attention mask, and the longest accepted branch wins.
- `pld k`: prompt lookup. The continuation (up to k-1 tokens) of the longest n-gram match (n=4..1) in the prompt plus finalized text. Its acceptance depends only on the greedy tokens, so it is simulated exactly from the greedy trace (no forward pass needed).
- Models:
  - Qwen3-8B-4bit: 48 new tokens; 7 code + 2 prose prompts, because mew's CPU was shared and the full set would have taken about 7 h.
  - Qwen3-1.7B-4bit: 64 new tokens on all 14 prompts, as a cross-check.
- coder-next was not run: the FlashRef CPU reference of an 80B MoE does not fit in mew's ~40 GB of free RAM at usable speed.
- Prompts are in `prompts/`: code completion (Rust, Python, TS, Go, SQL), code edits (rename, error handling, JS→TS), and write-a-function, plus 4 prose prompts as a control. Raw per-prompt TSVs are `q8b.tsv` and `q17b.tsv`. Run `summarize.py` to regenerate the tables.

## Results (tokens per weight read)
## q8b.tsv

| method | code tok/iter | prose tok/iter | all tok/iter | rows/token (all) | wasted rows | exact |
|---|---|---|---|---|---|---|
| greedy | 1.00 | 1.00 | 1.00 | 1.00 | 0% | yes |
| pld k=4 | 1.66 | 1.02 | 1.37 | 1.61 | 38% | yes |
| pld k=8 | 1.85 | 1.02 | 1.45 | 2.33 | 57% | yes |
| pld k=16 | 1.94 | 1.02 | 1.49 | 3.47 | 71% | yes |
| jacobi k=4 | 1.03 | 1.07 | 1.04 | 3.83 | 74% | yes |
| jacobi+pool k=4 n=3 G=2 | 1.31 | 1.09 | 1.23 | 3.26 | 69% | yes |
| jacobi k=8 | 1.03 | 1.07 | 1.04 | 7.67 | 87% | yes |
| jacobi+pool k=8 n=3 G=2 | 1.61 | 1.14 | 1.42 | 5.64 | 82% | yes |
| jacobi k=16 | 1.03 | 1.07 | 1.04 | 15.33 | 93% | yes |
| jacobi+pool k=16 n=2 G=2 | 1.85 | 1.23 | 1.58 | 10.11 | 90% | yes |
| jacobi+pool k=16 n=3 G=2 | 1.81 | 1.17 | 1.53 | 10.44 | 90% | yes |
| jacobi+pool k=16 n=4 G=2 | 1.63 | 1.12 | 1.41 | 11.33 | 91% | yes |
| jacobi+pool k=16 n=3 G=4 | 1.70 | 1.16 | 1.47 | 10.89 | 91% | yes |

## q17b.tsv

| method | code tok/iter | prose tok/iter | all tok/iter | rows/token (all) | wasted rows | exact |
|---|---|---|---|---|---|---|
| greedy | 1.00 | 1.00 | 1.00 | 1.00 | 0% | yes |
| pld k=4 | 1.87 | 1.02 | 1.51 | 1.69 | 41% | yes |
| pld k=8 | 2.13 | 1.02 | 1.63 | 2.60 | 62% | yes |
| pld k=16 | 2.27 | 1.02 | 1.68 | 4.18 | 76% | yes |
| jacobi k=4 | 1.05 | 1.06 | 1.05 | 3.79 | 74% | yes |
| jacobi+pool k=4 n=2 G=2 | 1.42 | 1.14 | 1.33 | 3.01 | 67% | yes |
| jacobi+pool k=4 n=3 G=2 | 1.41 | 1.08 | 1.30 | 3.08 | 68% | yes |
| jacobi+pool k=4 n=4 G=2 | 1.37 | 1.04 | 1.26 | 3.18 | 69% | yes |
| jacobi k=8 | 1.05 | 1.05 | 1.05 | 7.59 | 87% | yes |
| jacobi+pool k=8 n=2 G=2 | 1.67 | 1.22 | 1.51 | 5.29 | 81% | yes |
| jacobi+pool k=8 n=3 G=2 | 1.73 | 1.11 | 1.50 | 5.35 | 81% | yes |
| jacobi+pool k=8 n=4 G=2 | 1.62 | 1.06 | 1.41 | 5.69 | 82% | yes |
| jacobi+pool k=8 n=3 G=4 | 1.48 | 1.11 | 1.35 | 5.92 | 83% | yes |
| jacobi k=16 | 1.05 | 1.05 | 1.05 | 15.20 | 93% | yes |
| jacobi+pool k=16 n=2 G=2 | 1.89 | 1.32 | 1.68 | 9.50 | 89% | yes |
| jacobi+pool k=16 n=3 G=2 | 2.03 | 1.16 | 1.67 | 9.57 | 90% | yes |
| jacobi+pool k=16 n=4 G=2 | 1.80 | 1.08 | 1.52 | 10.55 | 91% | yes |
| jacobi+pool k=16 n=3 G=4 | 1.86 | 1.14 | 1.58 | 10.14 | 90% | yes |

`rows/token` is the compute multiplier over greedy: rows evaluated per finalized token. `wasted rows` is the fraction of evaluated rows that were never accepted.

## Where it stops being free
A window is free while decode stays memory-bound. With 4-bit weights, each row does about 3.6 flop per weight byte. On H200 (≈990 TFLOP/s dense bf16 against 4.8 TB/s) the ridge point is about 200 flop/B, so total rows × batch can reach roughly 55 (bf16), or about 110 with fp8 MMA, before compute becomes the limit. In practice kernels hit it earlier, around 30–50 rows.
- k=16 Jacobi+pool spends about 10 rows per token, so it is only free at batch ≤ 3–5, and it buys at most 1.6× tokens per read.
- Prompt lookup at k=4–8 spends 1.6–2.3 rows per token for 1.4–1.5× overall (1.7–1.9× on code), so it stays free up to batch ≈ 15–30.
- At serving batch sizes, Jacobi's wasted rows (74–93%) compete directly with other users' tokens.

## Verified vs inferred
- Verified (mew CPU, real weights): every number in the tables. Exact equality with greedy decoding held for every run.
- Inferred: the GPU break-even, from roofline arithmetic, not measured. The 8B prose numbers rest on only 2 prompts. Wall-clock times in the TSVs are from a heavily shared CPU and are not meaningful.
