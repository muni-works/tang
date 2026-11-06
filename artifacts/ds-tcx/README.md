# ds-tcx GPU validation

What is being validated: `DS_MOE_TCX=1` (the TCX grouped fp4 MoE, `frog/ds-tcx`). See
`docs/plans/ds-port-2026-11-05.md`.

Already done on mew's RTX 3090 (2026-11-05): NVRTC compiles the kernels, and
`gpu_tiny_moe_tcx_matches_ks` passes **bitwise** on the tiny fixture (fp4 experts, prefill,
6 decode steps, 3 verify windows). Speed has not been measured.

| script | needs | time | gate |
|---|---|---|---|
| `01_tiny_gates.sh` | 1 GPU, sm_80+, tiny fixture | ~10 min | TCX == K-split bitwise; full deepseek suite green with TCX 0 and 1 |
| `02_ep_gates.sh` | 2+ GPUs (NCCL) | ~20 min | ep KL with TCX on <= jitter-control KL (`DS_MOE_KS_S=2`) |
| `03_real_speed.sh` | 8 GPUs, real checkpoint (510 GB on disk) | ~45 min + weight download | decode tok/s TCX 1 > TCX 0 across 2 runs each; grouped_gu/down spans shrink |

Every script runs bash with `set -euo pipefail` and prints `DONE <name>` or `FAIL <name>`.
Logs go to `logs/`. Override `ROOT`, `CKPT`, `CARGO_TARGET_DIR`, `CUDARC_CUDA_VERSION` as
needed.

## GPU choice

- 01 runs on mew (3090) for free. Nothing to rent.
- 02 and 03 need every expert in VRAM: 272 GB of fp4 experts plus replicated attention,
  KV and Engram caches. The baseline (193 tok/s) was measured on **8x H100 80 GB**, so that
  is the setup to rent for an apples-to-apples speed number. **4x H200 141 GB** (564 GB)
  also fits at 68 GB of experts per rank, with fewer all-reduce hops, but it is not
  comparable to the baseline. 8x RTX PRO 6000 96 GB fits too, but it is PCIe-only (no
  NVLink), so the per-layer all-reduce would dominate. Not recommended.
- Recommendation: **8x H100 80 GB**, about 1.5 h total (weights download ~40 min at the
  RunPod container-disk rate, 02 ~20 min, 03 ~45 min). Put the weights on container disk,
  not a network volume.
