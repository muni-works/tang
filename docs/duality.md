# The bound and the tile are one LP

The claim being tested: the fastest kernel and the proof that it is
fastest are the dual and primal solutions of one linear program, read off
the comprehension's views, and the duality gap is the distance from
optimal. This note says what tang now computes, what it found, and where
the claim does not hold. Code: `crates/tang-compile/src/tensor_io_lp.rs`.
Experiment: `cargo run --release -p tang-bench --example io_dual`
(`TUNE=1` adds the GPU autotuner's pick on the local device). Lean:
`proofs/TangProofs/Duality.lean`, `Trace*.lean`.

## The LP

A kernel iterates a box `V = ∏_q [N_q]` (point dims, reduce dims, and the
reduce dims of every nested contraction) and reads arrays through views.
A pick view of array `j` is a projection `φ_j` onto a set of coordinates.
In exponents of a fast memory of `B` words, with `β_q = log_B N_q`:

```
dual  (tile):   max Σ_q x_q    s.t.  Σ_{q∈φ_j} x_q ≤ 1 − a_j    every operand tile fits
                                     x_q ≤ β_q                  no tile exceeds its extent
primal (bound): min Σ_j (1−a_j)s_j + Σ_q β_q t_q
                               s.t.  Σ_{j∋q} s_j + t_q ≥ 1      every coordinate covered
```

* **Primal → bound.** One stretch of `M` I/Os has `2M` words available and
  performs at most `∏_j (2M s_j/Σs / D_j)^{s_j} ∏_q N_q^{t_q} ∏_atomic N_q`
  body evaluations. That is the Hölder–Brascamp–Lieb inequality with loop
  bounds: Christ, Demmel, Knight, Scanlon and Yelick (2013), extended to
  arbitrary bounds by Dinh and Demmel (2020). So traffic is
  `≥ M·(|V|/F − 1)`.
* **Dual → tile.** The tile is `T_q = (M/J)^{x_q}`, with the budget split
  over the `J` operands.
* **The gap.** Strong duality makes the two optima equal: every solve
  returns both and checks `primal − dual ≈ 0` (`LpSolution::gap`). The
  remaining gap is between the bound's traffic and the dual tile's
  traffic, and it is a constant factor (below).

**Atomic coordinates.** A reduce dim of a nested contraction whose result
is consumed nonlinearly and never stored is *atomic*. Attention's head dim
`c` in `S = QKᵀ` is one, because `exp` needs the whole dot product. A
schedule must finish each such reduction inside one stretch, so `x_q = β_q`
and the arrays' budgets shrink by `a_j`. The alternative is to spill the
partial sums: then the inner result is one more array and `q` is free. A
schedule may pick either, so the bound is the minimum over the two models.

**Composite time floor.**
`TimeFloor = max(bytes / DRAM BW, flops / peak, launches × latency)`, with
the binding term named. `Machine::{RTX3090, M4}` carries the rates.

## Unit tests (`tensor::io_lp::tests`)

| test | result |
|---|---|
| `matmul_lp_is_hong_kung` | `σ = 3/2`, `s = x = (½,½,½)`, gap 0. Bound `M·(n³/(2M/3)^{3/2} − 1) ≈ 1.84·n³/√M` (Irony–Toledo–Tiskin form; COSMA's `2n³/√M` is ~9% stronger). Dual tile is a cube of side `√(M/3)`. |
| `attention_is_saha_ye` | For `M ≥ d²` the bound is a constant fraction of `N²d²/M` (scales ×4 in `N`, ×½ in `M`). For `M < d²` it is a constant fraction of `N²d/√M` (×½ for 4× `M`). This is Saha & Ye's `Θ(min(N²d²/M, N²d/√M))` (arXiv 2402.07443), from the LP alone. |
| `attention_dual_tile_is_flash_blocks` | The dual tile is `T_i = T_j = M/(4d)` = 48 for `M = 12288`, `d = 64`. Those are FlashAttention's `B_r = B_c = M/(4d)` (Dao et al., Alg. 1), derived, not given. |
| `small_matmul_hits_extent_bounds` | Short `k`: the extent bound binds (`t_k > 0`), gap 0. |
| `weak_duality_random` | 200 random projection LPs: primal feasible, gap < 1e-7. |
| `trace_is_current` | The Lean trace file equals what tang-compile exports today. |

## The experiment: dual tile vs what tang picks

All LPs below are read off the IR by `TensorGraph::kernel_lp` after tang's
pass pipeline. For attention that means the nested `S = QKᵀ` inside the
flash kernel, with its `c` marked atomic. Traffic is in MB at f32.

### Matmul

| device (M) | n | dual tile | dual traffic | `schedule_auto` tile | its traffic | autotuner (M4, timed) | its traffic | bound |
|---|---|---|---:|---|---:|---|---:|---:|
| 3090 (12288) | 1024 | 64×64×64 | 138 | 64×64 (k-chunk 32) | 138 | – | – | 71 |
| 3090 (12288) | 4096 | 64×64×64 | 8657 | 64×64 | 8657 | – | – | 4555 |
| M4 (8192) | 1024 | 52×52×52 | 172 | 64×64 | 138 | 32×64 (k 16) | 206 | 87 |
| M4 (8192) | 4096 | 52×52×52 | 10670 | 64×64 | 8657 | 64×64 (k 32) | 8657 | 5579 |

* **3090: match.** The dual's `√(M/3) = 64` is the output tile tang's
  heuristic picks (16×16 threads × 4×4 registers).
* **M4: tang's tile beats the dual.** The dual budgets `M/3` for each of
  A, B and C, but tang accumulates C in registers and stages only A and B
  k-chunks. So the real constraint is `T_i·T_k + T_k·T_j ≤ M` with C free,
  and a larger tile fits. The LP does not know which arrays live in
  registers. That is a modelling gap, not a refutation: with C removed from
  the shared-memory budget, the dual tile grows to `√(M/2)·…` and tang's
  64×64 sits inside it.
* **The autotuner optimizes time, not traffic.** It was timed on this M4
  (`TUNE=1`, wgpu/Metal). At 4096³ it keeps 64×64 (same as `schedule_auto`,
  1.55× the bound). At 1024³ it picks 32×64 with 16-wide k-chunks. That
  moves 1.5× more bytes than the dual tile (2.36× the bound) but makes
  512 workgroups instead of 256. At this size occupancy beats IO, and the
  time floor's memory term is not binding enough to matter.

### Attention forward (8 heads, N, d = 64, causal)

| device (M) | N | dual tile (q rows × k rows) | dual traffic | tang tile | tang traffic | autotuner | bound | floor @ bound | floor @ tang |
|---|---|---|---:|---|---:|---|---:|---|---|
| 3090 | 1024 | 48 × 48 | 96 | 16 × 16 | 273 | – | 11.1 | 0.030 ms compute | 0.291 ms memory |
| 3090 | 4096 | 48 × 48 | 1460 | 16 × 16 | 4312 | – | 179 | 0.483 ms compute | 4.61 ms memory |
| M4 | 1024 | 32 × 32 | 138 | 16 × 16 | 273 | 16 × 16 (273 MB) | 16.7 | 0.250 ms compute | 2.27 ms memory |
| M4 | 4096 | 32 × 32 | 2164 | 16 × 16 | 4312 | **64 × 32 (1091 MB)** | 268 | 4.00 ms compute | 35.9 ms memory |

* **The autotuner moves toward the dual and past it.** On the M4 at
  N = 4096, the timed autotuner picks 64 query rows × 32-key chunks. That
  is 4× tang's default and 1090 MB, which is *half* the dual tile's
  traffic (4.1× the bound). The equal `M/J` split again assumes that
  Q, K, V and O all share fast memory. The kernel keeps O (the online state)
  in registers, so Q (64×64) plus the K and V chunks (32×64 each) fill all
  8192 words. At N = 1024 it keeps 16 × 16: little reuse to win, so
  parallelism wins.
* **Default heuristic: mismatch, and the dual is closer.** The LP derives FlashAttention's
  block size from the views. tang's scalar flash kernel stages 16 query
  rows × 16-key chunks, which is 2.8× (3090) or 2× (M4) more traffic than
  the dual tile. The wider tiles are only found by the timed
  autotuner. The default `schedule_auto` / `nest_contractions` heuristic
  never picks them.
* **The time floor moves from compute to memory.** At the bound, attention
  is compute-bound on both devices. At tang's traffic it is memory-bound,
  with a floor 9–10× higher. That is the headroom a dual-sized tile would
  reclaim, if registers allow it (a 48-row tile of a 66-wide online state
  needs about 3.2K registers per row group).
* **The gap is constant.** The dual tile's traffic is 8.2–8.7× the bound
  at every N and M. The ratio is flat in N, so the dual tile is
  `Θ`-optimal and the 8× is the bound's constant: `N²d²/(4M)` from the
  atomic-unit count, against FA's achievable `≈ 4N²d²/M` causal.

## Coverage: what fits the LP

| view / op | fits? | what happens |
|---|---|---|
| affine pick view (transpose, broadcast, slice, head split `64h+2i` on separate coords) | yes | a projection |
| **guarded** pick view (zero padding) | yes | reads ⊆ the projection; the bound stays valid (padding only lowers the footprint) |
| multi-term form (`x[t + k]`, strided/dilated conv, polyphase) | **no** | HBL holds for general linear maps, but the LP's constraints then range over the lattice of subgroups (Christ et al. §6), which is exponential and not implemented. The array is *dropped* (`LpCoverage::non_projective`). That is sound (fewer arrays only weaken the bound) but loose: for a convolution the bound misses the input-reuse term. |
| **gather** view (`W[sel[t,k]]`, embeddings, paged KV) | **partly** | the address is data, so there is no projection. The gathered array is dropped from the LP (`LpCoverage::gathered`), and `Lp::gather_bound` adds it back as compulsory traffic, `max(bound(rest), Σ\|rest\| + U·slice)`, with `U` the distinct slices the index data touches. That is sound for the given data. `Lp::expected_distinct` gives `E(1 − (1 − k/E)^N)` for uniform top-k routing, which is #186's analytic decode floor. The reuse term of a grouped matmul (each expert slice against its tokens) still needs the per-expert token counts, and skewed routing changes it. |
| nested contraction consumed nonlinearly | yes, as *atomic* | the atomic/spill choice is the min of two LPs. `attention_bound` does both; `kernel_lp` from the IR builds the atomic model only. |
| causal / sliding masks (`select` on `iota`) | partly | the LP counts the full box. The bound is valid for tang's CDAG, which evaluates masked scores, but a mask-skipping schedule is a different CDAG and could halve it. |
| scans (`Op::Scan`), operators | no | not comprehensions; compulsory traffic only |
| tiny extents (`d < M^{x}`) | yes | extent constraints `x_q ≤ β_q` (`t_q > 0` in the primal) |

**Where constants and asymptotics break.**
* The primal's constant is HBL's. It is 1.84 vs 2 for matmul. For attention
  the atomic bound `N²d²/(4M)` is ~8× below what any known schedule
  achieves. That is the Θ-constant Saha & Ye leave unstated.
* The time floor uses peak rates and a fixed launch latency. The M4
  figures (120 GB/s, 4.3 TFLOP/s, 10 µs) are nominal, not measured.
* Registers are a level for the *tile* (`Lp::tile_levels`, below), but
  the *bound* uses one level at `regs + smem` words. That is sound and
  loses nothing at this scale. L2 is not modeled.

## Round 2: registers, bytes in flight, H200

### Two-level dual tile (`Lp::tile_levels`)

The single-level dual charges every operand to shared memory. tang (and any
GPU GEMM) keeps the accumulator in registers. The two-level dual puts the
output array (matmul's `C`, attention's online `O`) in registers
(`Machine::reg_words` per workgroup) and the inputs in shared memory
(`mem_words`). The footprints on each level must sum to at most its
capacity, and the innermost loop must run over a coordinate the output does
not depend on (the accumulator stays resident). The search is exhaustive
over power-of-two tiles and register/shared splits. The bound stays
single-level at `regs + smem` (sound, since on-chip storage is at most that).

`reg_words = 4096` is the schedule space's largest register tile (`reg` up
to 4×4 points × 256 threads). It is not the hardware register file.

| kernel (from the IR, `io_dual`) | single-level dual | **two-level dual** | M4 autotuner | traffic |
|---|---|---|---|---|
| matmul 4096³, M4 | 52×52 | **64×64**, C in registers | 64×64 | 8657 MB, the same as tang's |
| matmul 4096³, 3090 | 64×64 | **64×64** | (schedule_auto 64×64) | |
| attention fwd N=4096 d=64, M4 | 32 rows × 32 keys | **64 rows × 32 keys**, O in registers, Q/K/V in shared | **64×32** | 1090 MB vs 2164 MB, half, as the autotuner measured |
| attention fwd N=4096 d=64, 3090 | 48 rows | 128 rows × 32 keys, value columns split in two | not tuned | 830 MB vs 1460 MB |

So the register level **explains both M4 autotuner picks**. The "half the
dual's traffic" gap from round 1 is the online state moving out of shared
memory. Caveats:
* The match depends on `reg_words`. At 8192 words the matmul tile would
  be 128×64, which tang's space can't express.
* On the 3090 the IR LP leaves the value-column coordinate free and
  prefers splitting it (128×32, recomputing `QKᵀ` per half). The hand LP
  (`attention_lp`, `e` atomic) gives 64×64. Which one wins on hardware was
  not measured.
* Tests: `levels_match_m4_autotuner` (also checks bound ≤ two-level ≤
  single-level traffic).

### Bytes in flight (`Occupancy`, `TimeFloor::with_occupancy`)

Little's law: sustained bandwidth ≤ bytes in flight / latency, so a kernel
takes at least `bytes · latency / in_flight`. `in_flight` is resident
threads (from grid, block, registers per thread and the SM limits) × the
bytes one thread keeps outstanding. That term joins the composite floor and
is named `bytes-in-flight` when it binds. Machine constants:
* 3090: 82 SMs × 1536 threads, latency 470 ns, bandwidth-delay 0.44 MB.
* H200: 132 × 2048, latency 600 ns (assumed; not measured), 2.9 MB.

Launch floors are the smallest kernel in #186's traces: 2 µs on the 3090,
1 µs on the H200 inside a CUDA graph.

### Predicted vs measured: #186's kernels

From `scripts/ncu/predict_floor.py <#186's docs/bench/ncu> 96 64 16`. Bytes
are the 3090's ncu counts, and registers per thread come from the H200
trace. Per-thread in flight `b`:
* 96 B: the int-MMA Q4_K inner loop issues 8 + 24 + 64 B of independent
  loads per super-block before using them.
* 64 B: the lane GEMV's `uint4` pair plus scales.
* 16 B: one vector load, a calibration rather than a schedule value.

A ✗ marks a "floor" above the measured time (not a floor).

| kernel | GPU | measured µs / light | b = 96 (schedule) | b = 64 | b = 16 (calibrated) |
|---|---|---|---|---|---|
| MoE `ex_gate_up` T=1 (320×256) | 3090 | 22.1 / 0.71 | 15.7 / 1.00 memory | same | same |
| | H200 | 10.7 / **0.29** | 3.1 / 1.00 memory | 3.1 / 1.00 | 6.7 / 0.46 in-flight |
| MoE `ex_down` T=1 (160×256) | 3090 | 14.7 / 0.68 | 10.0 / 1.00 | same | same |
| | H200 | 5.4 / **0.36** | 1.9 / 1.00 | 2.1 / 0.91 in-flight | 8.5 / 0.23 ✗ |
| MoE `ex_gate_up` T=8 | H200 | 59.4 / **0.35** | 21.0 / 1.00 | 21.0 / 1.00 | 22.4 / 0.94 |
| MoE `ex_down` T=8 | H200 | 23.6 / **0.62** | 14.7 / 1.00 | 14.7 / 1.00 | 15.6 / 0.94 |
| MoE `ex_gate_up` T=48 | 3090 | 506 / 0.88 | 446 / 1.00 | same | same |
| | H200 | 182 / **0.48** | 87 / 1.00 | 87 / 1.00 | 116 / 0.75 in-flight |
| MoE `ex_down` T=48 | 3090 | 273 / 0.89 | 244 / 1.00 | same | same |
| | H200 | 125 / **0.38** | 47.5 / 1.00 | 47.5 / 1.00 | 63 / 0.75 in-flight |
| Q8_0 GEMV 256×256 | 3090 | 144 / 0.88 | 128 / 1.00 | same | same |
| | H200 | 63.5 / **0.39** | 24.9 / 1.00 | 24.9 / 1.00 | 68 / 0.36 ✗ |

### Predicted vs measured: H200 decode step

Group floors are #186's analytic bytes. A group's bytes are spread over its
launches by thread count, and each launch's floor is summed.

| config | wall ms | measured light | predicted light, b = 96 | b = 64 | b = 16 |
|---|---|---|---|---|---|
| hand N=1 | 4.04 | **0.18** | 0.68 | 0.65 | 0.44 |
| hand N=8 | 9.83 | **0.23** | 0.89 | 0.88 | 0.58 |
| hand N=48 | 29.67 | **0.28** | 0.95 | 0.94 | 0.37 |
| derived N=1 | 4.39 | **0.16** | 0.54 | 0.51 | 0.29 |
| derived N=8 | 10.91 | **0.20** | 0.77 | 0.75 | 0.46 |
| derived N=48 | 28.40 | **0.29** | 0.89 | 0.88 | 0.49 |

**Verdict: the new floor does not predict the H200.**
* **The 3090 is predicted.** Every big kernel is bandwidth-bound there at
  any `b ≥ 16`, as measured (0.7–0.9).
* **The H200 is not, with schedule-derived `b`.** With `b = 64–96 B` and
  unloaded latency, every big kernel has more than the 2.9 MB
  bandwidth-delay product in flight (82k resident threads × 64 B = 5 MB).
  Little's law then predicts light ≈ 1, against a measured 0.3–0.5.
  #186's diagnosis ("too few bytes in flight") is not reproduced by
  counting threads × issued loads.
* **The class flip comes out only with `b ≈ 16 B`.** Then the 3090 stays
  bandwidth-bound while the H200 turns in-flight-bound. That is
  qualitatively #186's finding, but the per-kernel error is up to 2.3×,
  and four kernels predict a time above the measured one. So `b ≈ 16 B` is
  a fit, not a floor.
* **The step: launches explain part of N=1, nothing explains the rest.**
  The launch term (~550–830 launches × 1 µs) accounts for a share of the
  N=1 gap: predicted 0.51–0.68 vs measured 0.16–0.18. At N=48 the floor
  is within 10% of pure bandwidth, while the step runs 3.4× slower.

What the model is missing:
* **Loaded latency.** At 50% of peak, HBM latency is well above 600 ns.
* **Effective outstanding requests.** Dependent dequant chains, the
  per-SM miss queue and L1 MSHR limits all cut it below what the threads
  issue. tang can't see this, and it needs per-SM counters (ncu on the
  H200, unavailable in RunPod containers).
* **Tails.** Gate_up T=1 is ~2.8 load rounds per thread.
* **Bytes.** The H200 light uses the 3090's bytes; the 50 MB L2 changes them.

The next measurement that would separate these: ncu
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss` and
`lts__average_gcomp_input_sector_*` on a host with H100/H200 counters.

## Lean

* **`Tang.Duality.weak_duality`.** For any covering HBL exponents `s` and
  any fitting tile exponents `x`, `Σ x ≤ Σ s`. LP weak duality for this
  pair, mechanized.
* **`box_hbl`, `box_volume`.** For a box tile with sides `T_q ≥ 1`,
  `∏ T ≤ ∏_j |φ_j(E)|^{s_j}`, and it is `≤ M^{Σs}` when every footprint is
  `≤ M`.
* **`tiles_needed`.** `n` tiles of volume `≤ M^σ` covering `V` points
  number at least `V/M^σ`. Together these give **no box-tiled schedule
  moves fewer words than the primal**.
* **Not mechanized.** The same statement for arbitrary schedules, which
  needs the discrete HBL inequality for non-box sets (Loomis–Whitney in
  the matmul case). It is not in Mathlib, and proving it is a project of
  its own.
* **`Tang.Trace.combine_eq_osm`.** The combine of the online-softmax
  monoid that tang's `stabilize_softmax` actually built for
  `Σ e^{s−m} v / Σ e^{s−m}` is checked in Lean. It is exported node for
  node from its tang-expr graph into `Trace.lean` (generated; the Rust test
  `trace_is_current` fails if it drifts). Lean proves it equals `OSM.combine`,
  so the real rewrite inherits associativity, tiling and flash = two-pass
  softmax. Caveat: the trace's `log₂ e` literal is read as the real number.
  The f64 constant differs in the last bit, and `Select` at `a = b` picks
  `b` (= `max`).
