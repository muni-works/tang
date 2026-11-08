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
| **gather** view (`W[sel[t,k]]`, embeddings, paged KV) | **no** | the address is data, so there is no projection. Dropped (`LpCoverage::gathered`), and the compulsory bound caps touched slices instead. The per-expert grouped-matmul bound needs a routing assumption (balanced) the LP cannot see, and a skewed routing changes its asymptotics. |
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
* Registers are not a level in the LP. A tile the dual allows may not fit
  the register file, and a tile tang keeps in registers (matmul's C) is
  charged to shared memory.
* One level only (shared memory ↔ DRAM). L2 and the register file are the
  same LP at other `M`s, but nothing composes the levels yet.

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
