# FP8 on Hopper (H200, 2026-11-06)

RunPod community H200 SXM (700 W, 1980 MHz). tang: `examples/hopper_fp8.rs`,
`examples/hopper_fp8_attn.rs`; baselines: `scripts/hopper_attention.py fp8mm |
matmul | fp8attn` (cuBLASLt via `torch._scaled_mm`, FA3 from
`kernels-community/flash-attn3`). Inputs zero-mean, unit variance. Error is
relative to the largest |reference|, on 256 sampled entries (matmul) or two
heads (attention); "predicted" is the emulator, computed before the kernel
runs and checked against `TANG_BUDGET`.

## Matmul `C = A·Bᵀ`, A [m,k], B [n,k], bf16 out (TFLOP/s)

| shape | tang E4M3 1×128 blocks | tang E4M3 tensorwise, f32 accum | tang E4M3 tensorwise, `Fp8Accum` | tang bf16 wgmma | cuBLASLt FP8 tensorwise | cuBLASLt FP8 rowwise | cuBLAS bf16 |
|---|---|---|---|---|---|---|---|
| 2048³ | 401 | – | – | 336 | 1041 | 841 | – |
| 4096³ | 556 | 569 | 794 | 399 | 1360 | 1134 | 789 |
| 8192³ | 731 | 757 | 1013 | 430 | 1457 | 1045 | 707 |
| 16384×2048×8192 | 549 | 545 | 692 | 387 | 1203 | 892 | – |
| 16384×8192×2048 | 694 | over budget | 989 | 497 | 1378 | 1043 | – |

E5M2 1×128: 561 (4096³), 735 (8192³).

Error, E4M3 1×128 (512×1024×384, whole graph through `precision_error`):
emulated vs f64 max 3.17e-2 / mean 4.3e-4; GPU vs f64 3.22e-2 / 4.3e-4; GPU vs
emulated 4.1e-3 (bf16 output rounding). With f32 output the GPU matches the
emulator to ~2e-4 at every size (accumulation order).

`Fp8Accum` (no promotion): per-addition truncation to 14 bits matched an H100
on biased data to 4e-4. On the H200 with zero-mean data, truncation once per
k32 instruction (≥14 bits) matches to the f32 floor (1.1e-3) while the
per-addition model over-predicts max error ~4× (8e-2 vs 2.5e-2 at 4096×8192).
The emulator keeps the per-addition model: conservative. Refining it to
per-instruction is a follow-up.

Precision search (`MM_TUNE=1024x2048x1024`, f32 0.068 ms): at budget 3e-2
and 3e-3 E4M3 is predicted 3.3e-2, never run, bf16 kept (0.028 ms). On an
H100 with a narrower input range it picked E4M3 (0.0167 ms vs bf16 0.0296)
at 3e-2 and rejected it (predicted 7.57e-3, measured 7.55e-3) at 3e-3.

## Attention forward, d=128, 16k tokens × 16 heads (TFLOP/s)

| n | tang E4M3 1×128 | tang E4M3 tensorwise | FA3 FP8 | FA3 bf16 | SDPA bf16 |
|---|---|---|---|---|---|
| 1024 | 333 | 359 | 743 | 633 | 334 |
| 2048 | 397 | 433 | 797 | 682 | 348 |
| 4096 | 438 | over budget* | 856 | 705 | 342 |
| 8192 | 462 | 518 | 898 | 715 | 346 |
| 16384 | 475 | 535 | 927 | 698 | 350 |

Causal (tensorwise): tang 228 / 297 / 350 / 386 / 433; FA3 FP8 518 / 616 /
819 / 829 / 889.

\* predicted max_rel 1.25e-1 > budget 0.1: not run.

Error (E4M3 1×128): predicted max ~6–8e-2, mean ~8e-3; small graph (bh=2,
n=256) through `precision_error`: emulated 5.78e-2 / 8.3e-3, GPU 5.71e-2 /
8.4e-3, GPU vs emulated 2.7e-2 / 2.9e-3 (P is rounded at block maxima on the
GPU, at running maxima in the emulator: same statistics, different
roundings). FA3 FP8: max 7.4–9.2e-2, mean 5.7–8.2e-3. At budget 5e-2 the
schedule is rejected before it runs.

## Where the gap is

- Matmul: tang's FP8 tile reaches ~50% of cuBLASLt tensorwise (70% with
  fast accumulation). The producer is `cp.async` (H100 profile: with copies
  off the consumers alone reach ~600 vs ~300 with them before the copy loop
  was hoisted; now ~730). TMA (needs a host tensor map) and 2-CTA clusters
  are the next steps.
- Attention: ~55–60% of FA3 FP8. P goes through shared memory (two named
  barriers per step) instead of an in-register byte permute, and there is no
  intra-warpgroup overlap or ping-pong between consumer warpgroups.
