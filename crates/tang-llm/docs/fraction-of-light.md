# Fraction of light: measured kernels against a traffic floor (v0)

This document compares each kernel's measured time with the time its HBM traffic would take
at peak bandwidth (the *floor*): **light = floor / measured**. A light of 1.0 means the kernel
moves its bytes at peak; 0.2 means it runs five times slower than its traffic requires.

This is v0: the floor counts the bytes the kernel actually moved (ncu `dram__bytes_*`), or,
for whole decode steps, the analytic weight, state and KV bytes a step has to read. The
derived lower bound from the Prove session replaces both later.

## Harness (`scripts/ncu/`)

| script | what it does |
|---|---|
| `ncu_kernels.sh OUT.csv [ncu opts] -- cmd` | per launch: `dram__bytes_read.sum`, `dram__bytes_write.sum`, `gpu__time_duration.sum`, `sm__throughput` and `dram__throughput` (% of peak), grid and block. Warm cache, boost clocks (`--cache-control none --clock-control none`). |
| `table.py CSV --bw B` | per kernel: launches, µs each, MB each, floor, light, SM %, DRAM %, and a bound class (below); `--step-ms` adds the launch/idle gap |
| `nsys_kernels.sh` / `nsys_table.py` | for GPUs whose counters ncu can't read (RunPod containers return `ERR_NVGPUCTRPERM`): kernel durations and inter-kernel idle time from an nsys trace (CUDA graph nodes included), with bytes joined from an ncu run of the same workload on another GPU |
| `step_groups.py` | one decode step grouped into moe / dense / gdn / attn / small, each against its analytic floor |
| `step_floor.py MODEL.gguf N…` | the step's weight bytes from the GGUF: non-expert tensors once, plus the distinct experts touched, E·(1 − (1 − k/E)^N) per layer |
| `run_all.sh` / `run_nsys.sh` | the whole sweep (ncu or nsys) |
| `cudarc_stubs.c` | `LD_PRELOAD` stubs for driver/NVRTC symbols that cudarc resolves up front and some ncu injections lack |

Engine hooks added for profiling:
- `TANG_BATCH_ONLY=b` runs only that batch size in `flash-batch-check`.
- `TANG_PROFILE_STEPS=k` runs k steps inside `cuProfilerStart`/`Stop` (ncu `--profile-from-start off`, nsys `--capture-range=cudaProfilerApi`).
- `ONCE=1` in `coder_derived_moe` / `coder_derived_ops` launches three times, with no timing loops.
- Derived kernels are named after their graph node (`dop08_ex_gate_up`), not `main0`.

**ncu version matters.** cudarc 0.16 resolves every symbol when it loads. Nsight Compute
2025.1 lacks `cuTensorMapEncodeIm2colWide`, and 2026.4's NVRTC interception lacks
`nvrtcGetNVVM`. **2025.2.1 works** (installed on mew under `/opt/nvidia/nsight-compute/`).

**Bound classes**, per kernel's mean launch:

| class | condition | meaning |
|---|---|---|
| launch | under 5 µs | fixed launch, ramp and tail cost dominates |
| traffic | DRAM ≥ 50% of peak, or light ≥ 0.5 | limited by bytes moved |
| compute | SM ≥ 60% of peak | limited by arithmetic |
| latency | none of the above | fills neither DRAM nor the SMs: too few bytes in flight (too few CTAs, dependent loads, serial reductions) |

## Hardware runs

- **RTX 3090** (mew, `flock ~/tang-gpu.lock`): ncu counters, 936 GB/s. Tables in `docs/bench/ncu/3090/`.
- **H200 SXM** (RunPod secure, 4.8 TB/s): no counter access in the container, so the nsys path was used, with bytes from the 3090 ncu run for the same kernels and shapes. Tables in `docs/bench/ncu/h200/`; raw traces in `h200-raw/`. The pod ran about 50 minutes (roughly $4.4).

## The decode step (H200, Coder-Next UD-Q4_K_XL, context ~1380, CUDA graphs)

Group times come from one profiled step and are expressed as a share of that run's wall time per step.

| config | wall ms | moe light | dense light | gdn light | attn light | small (<5 µs) | idle between kernels | **step light** |
|---|---|---|---|---|---|---|---|---|
| hand N=1 | 4.04 | 0.14 (40%) | 0.37 (31%) | 0.13 | 0.02 | 15% | 1% | **0.18** |
| hand N=8 | 9.83 | 0.27 (55%) | 0.18 (26%) | 0.43 | 0.08 | 7% | 0% | **0.23** |
| hand N=48 | 29.67 | 0.36 (56%) | 0.08 (21%) | 0.46 | 0.13 | 4% | 0% | **0.28** |
| derived N=1 | 4.39 | 0.17 (30%) | 0.40 (26%) | 0.14 | 0.02 | 25% | 8% | **0.16** |
| derived N=8 | 10.91 | 0.25 (54%) | 0.18 (23%) | 0.48 | 0.08 | 9% | 4% | **0.20** |
| derived N=48 | 28.40 | 0.42 (51%) | 0.08 (21%) | 0.49 | 0.13 | 7% | 2% | **0.29** |

The analytic floors behind the light column:

| N | step floor | weights | GDN state | KV | distinct experts per layer |
|---|---|---|---|---|---|
| 1 | 3.46 GB, 0.72 ms | 3.28 GB | 0.15 GB | 0.03 GB | 10 |
| 8 | 10.7 GB, 2.23 ms | 9.21 GB | | | 75 |
| 48 | 39.9 GB, 8.32 ms | 31.1 GB | 7.2 GB | 1.6 GB | 313 |

Weights are 2.36 GB of non-expert tensors plus 1.909 MB per expert. GDN state is 36 layers × 2 MiB f32 per stream, read and written. Per-config tables are in `docs/bench/ncu/h200/step_*.md`.

**Inside a CUDA graph the step is not bound by launch gaps.** Idle time between kernels is 0–1% for the hand engine and 2–8% for derived (which runs more, smaller kernels). The ~0.5–1 ms of kernels under 5 µs at N=1 is real kernel time (norms, quantizers, routing), not gaps.

## Per kernel: the same code on two GPUs

| kernel (derived) | 3090: µs / light / DRAM % | H200: µs / light | class on H200 |
|---|---|---|---|
| MoE `ex_gate_up`, T=1 (320 CTAs) | 21.8 / 0.72 / 74% | 10.7 / 0.29 | latency |
| MoE `ex_down`, T=1 (160 CTAs) | 14.8 / 0.68 / 70% | 5.4 / 0.36 | latency |
| MoE `ex_gate_up`, T=48 | 506 / 0.88 / 90% | 182 / 0.48 | latency |
| MoE `ex_down`, T=48 | 273 / 0.89 / 92% | 125 / 0.38 | latency |
| Q8_0 GEMV, 2048 → 4096 rows (256 CTAs) | 144 / 0.88 / 70% | 63.5 / 0.39 | latency |
| GDN chunked scan, H=32, T=4096 (2 kernels) | 690 / 0.63 / 61–66% | 434 / ≈0.2 | latency (one kernel tuned to 4 CTAs) |
| shared expert GEMVs, T=8 | 13 / 0.10 / 11% | 12 / 0.03 | latency |
| top-k, norms, quantizers | 2–12 / under 0.05 | 1–9 | launch / latency |

The H200 light figures assume the kernel moves the same bytes on both GPUs (the 3090's ncu bytes). That holds up to L2 effects: the H200's L2 is 50 MB against the 3090's 6 MB.

On the 3090 the big derived kernels reach 0.7–0.9 of light: there they are traffic-bound and close to done. On the H200 the same kernels reach only 0.3–0.5. The H200 has 5.1× the bandwidth but the kernels run just 2–3× faster. They keep too few bytes in flight to cover the H200's bandwidth-delay product, which is about 5× larger. So the same code changes class, from traffic-bound to latency-bound (memory-level parallelism).

Flash attention fwd/bwd on the H200: `hopper_attention` autotunes inside the run, so the trace mixes candidate variants. Its printed best-of times were not kept, so this run gives no clean H200 number. On the 3090, the fused attention backward's two main kernels run at 24–32% SM and 4–6% DRAM, classed latency. Attention is compute-shaped (4·N²·D FLOPs against 4·N·D·2 bytes), so its floor should be the FLOP roof, not traffic. Rerunning with `TANG_VARIANT` fixed is the follow-up.

## What it implies for the 1–8 stream gap

1. **The gap isn't launches; it's bytes in flight.** At N=1 the MoE group is 40% of the hand step at light 0.14, and dense GEMVs are 31% at 0.37. The expert GEMVs run 64–320 CTAs per call on 132 SMs, each making one pass over a row slice. Each call is too short to ramp to full bandwidth before its tail, and 10 experts × 3 matrices per layer pay that ramp and tail every time.
2. **GEMV split / more parallelism per byte.** Splitting K across CTAs (or more rows per warp with deeper `cp.async` / TMA prefetch) raises outstanding loads per SM. The 3090 numbers show the inner loop is already right at 0.88 of light, so the missing piece is occupancy and pipelining on the H200, not the arithmetic. This alone targets the 0.14 → ~0.5 range for the MoE group.
3. **Persistent fused MoE kernel.** One resident kernel per layer that streams gate|up, then down, for every (token, expert) pair keeps loads in flight across expert and kernel boundaries. It removes the per-call ramp and tail (the main cost at N=1–8) and absorbs the ~15–25% of step time spent in sub-5 µs kernels (route, norms, quantize, combine). That is where derived is behind hand at N=1–8 (more small kernels, 8% idle against 1%).
4. **Dense projections at N≥8** are the other large lever: light 0.18 at N=8 and 0.08 at N=48. `cx_mma_q8x` takes 4 ms per step at N=48 against a 0.46 ms weight floor, with only 192 CTAs. Same disease, same cure.
5. **GDN state traffic is real at N=48:** 7.2 GB per step, about 1.5 ms of floor, at light 0.46–0.49. State in bf16 (`TANG_CODER_GDN_STATE=bf16`) would halve that floor.
