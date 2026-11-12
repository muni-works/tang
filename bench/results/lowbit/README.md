# Low-bit weights for Qwen3-8B

Can tang serve Qwen3-8B on fewer than MLX's 4.5 bits per weight, keep
perplexity within 3.69 (MLX 4-bit: 3.651; bf16: 3.515), and decode faster?

**Answer, 2026-11-11:**
- **Quality: yes, just about.** GPTQ 3-bit, group 64, behind a block-Hadamard rotation (`g3g64hs`) is 3.5 bpw. Its ppl is 3.6915 and its greedy agreement with bf16 is 0.956. The weights file is 22% smaller (3.74 GB vs 4.61 GB).
- **Speed: not yet.** On an H200, the derived decode step is 0.78× a4. Without its rotation kernels it is 0.94× a4. The decode GEMV runs at about 1 TB/s, a fifth of the H200's bandwidth, so cutting bytes doesn't cut time until the GEMV is bandwidth-bound.

## Quality

Measured with `bench/lowbit/lowbit.py`, the torch twin of `tensor_lowbit.rs`:

- **Perplexity:** 4096 tokens, half code from `tang-la` (Cholesky, QR, LU) and half prose (README, NOTES, CONTRIBUTING).
- **Agreement:** greedy-token agreement on 10 coding prompts × 32 tokens, each continuation teacher-forced on the bf16 (or a4) greedy output.
- **Coverage:** only the decoder-block projections change; the embedding and LM head stay MLX 4-bit.
- **Calibration (GPTQ):** 128 windows of 256 tokens from tang's own sources, none from the eval text.
- **Hessians:** taken at each block's q/o/gate/down inputs on the bf16 model.

| format | bpw | ppl | agree vs bf16 | agree vs a4 |
|---|---|---|---|---|
| bf16 | 16 | 3.515 | 1 | — |
| MLX 4-bit (`a4`, g64) | 4.5 | 3.651 | 0.959 | 1 |
| RTN 4-bit g128 + Hadamard (`a4g128h`) | 4.25 | 3.776 | 0.966 | 0.938 |
| RTN VQ 2-d, 128 codes + H (`vq2d128h`) | 3.625 | 4.098 | 0.928 | 0.916 |
| RTN 3-bit g64 + H (`a3g64h`) | 3.5 | 4.418 | 0.894 | 0.900 |
| RTN 3-bit g64 (`a3g64`) | 3.5 | 4.793 | 0.884 | 0.884 |
| RTN VQ 4-d, 2048 codes + H (`vq4d2048h`) | 2.875 | 4.567 | 0.884 | 0.859 |
| RTN 2-bit g64 + H (`a2g64h`) | 2.5 | 40420 | 0.028 | 0.031 |
| **GPTQ 3-bit g64 + H, rotation per matrix (`g3g64h`)** | 3.5 | **3.684** | 0.947 | 0.931 |
| **GPTQ 3-bit g64 + H, rotation per input (`g3g64hs`)** | 3.5 | **3.6915** | 0.956 | 0.922 |
| GPTQ 3-bit g128 + H (`g3g128h`) | 3.25 | 3.766 | 0.953 | 0.950 |
| GPTQ 3-bit g64, no rotation (`g3g64`) | 3.5 | 3.831 | 0.903 | 0.909 |

What the table shows:
- **Round-to-nearest isn't enough below 4 bits.** At equal bits, VQ beats scalar (2.875-bit VQ ≈ 3.5-bit scalar), but neither gets close to the budget.
- **GPTQ recovers most of the gap.** 3-bit g64 drops from 4.42 to 3.68.
- **GPTQ needs the Hadamard rotation.** Without it, 3-bit GPTQ lands at 3.83.
- **`g3g64hs` is the format the graph runs.** Its rotation is seeded per projection input, so q/k/v share one rotation and gate/up share one. A graph applies each once, at four sites per layer. Its ppl is 0.0015 over the budget, on a 4096-token eval; call it a tie.

## Speed: H200, derived decode step (`scripts/derived-race.sh` via the runner)

| weights | kernels/step | len 128 | len 4096 | greedy ids |
|---|---|---|---|---|
| MLX 4-bit | 581 | 3.925 ms (254.8 tok/s) | 4.277 ms (233.8 tok/s) | hand = derived |
| g3g64hs (with rotation) | 725 | 5.062 ms (197.6 tok/s) | 5.420 ms (184.5 tok/s) | hand = derived; same as a4 for the first 32 of 64 |
| g3g64hs, rotation skipped (timing only) | 581 | 4.169 ms (239.9 tok/s) | 4.519 ms (221.3 tok/s) | — |

Hand and derived step times agree to within 2% for both weight formats.

Why g3 is slower today:
1. **Rotation launches:** the rotation adds 144 kernels per step (4 sites × 36 layers), about 1.1 ms. The passes don't fuse a 128-wide reduction into its neighbours.
2. **GEMV speed:** even without the rotation, the U3 GEMV is 6% slower than U4. The a4 step streams ~3.9 GB of weights in 3.9 ms, about 1 TB/s against the H200's 4.8 TB/s. The GEMV is limited by instructions and latency, not bytes. U3 has the same per-weight decode work as U4, plus a little extra.

**Projected speedup:** the byte ratio is the ceiling. The weights a step streams drop from ~3.9 GB to ~3.1 GB (projections at 3.5 instead of 4.5 bpw), about **1.27×**, once two things hold:
- **(a)** the decode GEMV is bandwidth-bound, for both formats;
- **(b)** the rotations are free. Fold them offline QuaRot-style (a rotated residual stream; fold the o-site rotation into v_proj), which leaves only the down-proj input rotation online, fused into its producer.

## Standalone kernels (`bench/lowbit/vq_bench.cu`)

DRAM numbers: every timed call reads a different copy of its weights, spanning more than 4× the L2.

| shape | bf16 | a4 | g3 (incl. rotate + group sums) | vq |
|---|---|---|---|---|
| A100 4096×4096 | 25.2 µs | 21.8 µs | 28.5 µs | 55.6 µs |
| A100 12288×4096 | 64.4 µs | 43.9 µs | 57.3 µs | 105.7 µs |
| A100 4096×12288 | 65.6 µs | 49.0 µs | 62.8 µs | 139.1 µs |
| H200 4096×4096 | 13.0 µs | 11.9 µs | 15.8 µs | 20.4 µs |
| H200 12288×4096 | 28.6 µs | 24.0 µs | 30.0 µs | 42.7 µs |
| H200 4096×12288 | 31.7 µs | 25.1 µs | 32.7 µs | 48.0 µs |

Here too the packed kernels are instruction-bound, at 0.4–1.2 TB/s, while bf16 reaches 1.3–3.5 TB/s. VQ's codebook gathers make it the slowest, which is why VQ is out.

## Reproduce

```
# CPU box with ~32 GB free (bf16 weights in ~/models/Qwen3-8B, MLX 4-bit in the HF cache)
python3 bench/lowbit/lowbit.py texts
python3 bench/lowbit/lowbit.py refs
python3 bench/lowbit/lowbit.py calib calib.txt 128 256 0 12   # then 12 24, 24 36 (Hessians ~28 GB)
python3 bench/lowbit/lowbit.py fmt g3g64hs                    # GPTQ runs on a GPU when there is one
python3 bench/lowbit/lowbit.py export g3g64hs /models/qwen3-8b-g3

# H200, through the warm runner
scripts/gpu-run.sh h200-1 claude/lowbit-g3 race:qwen3-8b LENS=128,4096
scripts/gpu-run.sh h200-1 claude/lowbit-g3 'cmd:MODEL_DIR=/models/qwen3-8b-g3 OUT=/root/lowbit-g3-race LENS=128,4096 bash scripts/derived-race.sh'

# standalone kernels (any CUDA GPU)
D=$PWD/bench/lowbit; VQ_CU_OUT=$D/vq.cu VQ_CU_DIR=$D cargo test -p tang-compile --lib _dump -- --ignored
cd bench/lowbit && nvcc -O3 -arch=native -I . vq_bench.cu -o /tmp/vqb && /tmp/vqb
```
