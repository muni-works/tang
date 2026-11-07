# What tang proves

Two kinds of guarantee back the claim that tang's derived kernels can
replace hand kernels:

1. **The rewrites are sound.** Fusion, tiling/splitting a reduction, the
   online-softmax (flash) rewrite, max-stabilization and derived gradients
   preserve the function computed. These are mechanized in Lean 4 +
   Mathlib in [`proofs/`](../proofs) (`lake build`, no `sorry`).
2. **The schedules are measured against IO lower bounds.** For every
   kernel class, `tang_compile::tensor::io` computes a red–blue-pebbling
   lower bound on slow↔fast-memory traffic over *all* schedules of the
   comprehension graph, and the traffic of the schedule tang picks;
   `cargo run --release -p tang-bench --example io_bounds -- --md` prints
   the table below.

Everything is over the reals. Floating point is not modelled: the
rewrites reassociate sums and rescale by `exp(m − m')`, which is exact in
ℝ and changes rounding in f32 (see [Not proven](#what-is-not-proven)).

## Part I — soundness (mechanized)

Lean names refer to `proofs/TangProofs/*.lean`.

**Model.** A tensor of rank `s` is a function `ℤˢ → α`. A *form* over an
`r`-dimensional space is `f(i) = b + Σ_d a_d i_d`; a *view* is one form per
source dim; reading `x` through view `v` at point `i` is `x(v(i))`. A
*comprehension* is `out(p) = ⨁_{k∈K} body(a(pt(p,k)))` for a commutative
monoid `⨁` (sum; max and min on `WithBot`/`WithTop`; product). This is
`Form`, `View`, `Comp` in `crates/tang-compile/src/tensor.rs`.

**Theorem 1 (view composition; `Form.eval_compose`, `View.read_compose`).**
For forms `f` over an `s`-dim space and `inner : Fin s → Form r`,
`(f ∘ inner)(i) = f(inner(i))`, where `f ∘ inner` is computed
coefficient-wise exactly as `Form::compose` does
(`coefs_e = Σ_d a_d · inner_d.coefs_e`, `offset = b + Σ_d a_d · inner_d.offset`).
Consequently reading through the composed view equals reading through
both.
*Proof.* Expand, distribute, exchange the double sum (`Finset.sum_comm`). ∎

**Theorem 2 (fusion soundness; `substitute_map_sound`, `substitute_map_sound₂`).**
Let `m(j) = f(x(v(j)))` be a map and `C` a comprehension reading `m`
through view `w` (and possibly other tensors). Replacing the read of `m`
by `f` applied to `x` read through `v ∘ w` — what `substitute_maps`
does — leaves `C` unchanged, for every monoid and body.
*Proof.* By Theorem 1, `w.read (f ∘ v.read x) = f ∘ (v ∘ w).read x`
pointwise; the comprehension is a congruence. ∎

**Theorem 3 (split / tiled reductions; `split_reduction`).** For any
monoid and any partition of an ordered reduction into consecutive chunks,
reducing each chunk and then the partials equals reducing the whole.
This covers `split_reduction`, `schedule_split_k`, workgroup tree
combines and chunked staging; commutativity is not needed.
*Proof.* `List.prod_flatten`. ∎

**Definition (online softmax).** State `(m, l, o) ∈ ℝ³`;
`(m₁,l₁,o₁) ⊕ (m₂,l₂,o₂) = (M, l₁e^{m₁−M} + l₂e^{m₂−M}, o₁e^{m₁−M} + o₂e^{m₂−M})`
with `M = max(m₁, m₂)`; a score `x` with value `v` is `(x, 1, v)`. A
`d`-vector of values is `d` copies of the scalar channel sharing `m, l`.

**Theorem 4 (semantics; `OSM.sem_combine`, `OSM.eq_of_sem`).** With
`⟦(m,l,o)⟧ = (l·eᵐ, o·eᵐ)`, `⟦a ⊕ b⟧ = ⟦a⟧ + ⟦b⟧`; a state is determined
by `m` and `⟦·⟧`.

**Theorem 5 (online softmax is a commutative semigroup;
`OSM.combine_assoc`, `OSM.combine_comm`).**
*Proof.* Both sides have equal `m` (`max` is associative/commutative)
and equal meaning (Theorem 4 and associativity of `+`); apply
`eq_of_sem`. ∎
So by Theorem 3 any tiling, split or reordering of the key loop computes
the same state.

**Theorem 6 (flash = two-pass softmax; `OSM.run_eq_softmax`,
`OSM.run_append`).** For any nonempty sequence of scores `xᵢ` with values
`vᵢ`, folding `⊕` left to right yields `m = maxᵢ xᵢ` and
`o / l = Σᵢ e^{xᵢ} vᵢ / Σᵢ e^{xᵢ} = Σᵢ softmax(x)ᵢ vᵢ`; and the fold of a
concatenation of two tiles is `⊕` of the tiles' folds.
*Proof.* Induction on the list: `⟦fold⟧ = Σ (e^{xᵢ}, e^{xᵢ}vᵢ)`, the
`m`-component is the running max; divide, cancelling `eᵐ`. ∎

**Theorem 7 (shift invariance; `softmaxShift_eq`, `lse_shift`).** For
every `c ∈ ℝ`, `e^{xᵢ−c}/Σⱼ e^{xⱼ−c} = softmax(x)ᵢ` and, for `n > 0`,
`c + log Σⱼ e^{xⱼ−c} = log Σⱼ e^{xⱼ}`. This is what licenses
`stabilize_softmax` to subtract the row max.

**Theorem 8 (gradients of comprehensions are comprehensions;
`vjp_sum_comp`, `vjp_fiber`).** Let `yᵢ = Σ_k h_{ik}(x(v(i,k)))` with
differentiable bodies (`h_{ik}` closes over the other arguments, so
multi-argument bodies are included) and `L = Σᵢ gᵢ yᵢ`. Then
`∂L/∂x_a = Σ_{i,k} [v(i,k) = a] · gᵢ · h'_{ik}(x_a)`
— a sum comprehension over the *same* iteration space — and summing out
the gate, `= Σ_{(i,k) ∈ v⁻¹(a)} gᵢ h'_{ik}(x_a)`, a sum over the fiber
of the view. For a pick view the fiber is a box (an ordinary
comprehension over the eliminated coordinates); for a gather through a
sorted index it is a run (`TNode::segment`).
*Proof.* `HasDerivAt` of each term: the bump changes `x(v(i,k))` iff
`v(i,k) = a`; sum and scale. ∎

**Theorem 9 (the stabilizing max has zero gradient;
`stabilized_softmax_eq`, `stabilized_softmax_hasFDerivAt`,
`softmaxShift_deriv_shift`, `lse_deriv_shift`).** For *any* shift
function `c : ℝⁿ → ℝ` — in particular `c = max`, which is not
differentiable at ties — `x ↦ softmaxShift(x, c(x))` *is* `softmax`, so it
has exactly softmax's Fréchet derivative everywhere; and the partial
derivative with respect to the shift is `0`. Same for log-sum-exp.
Hence the backward never needs `∂max/∂x`: `tensor_grad.rs`'s elimination
of the max's gradient is sound, including at ties.
*Proof.* Theorem 7 gives equality of functions; derivatives of equal
functions are equal. ∎

**Theorem 10 (flash backward identity; `softmax_avg_deriv`).** With
`y = Σᵢ softmax(x)ᵢ vᵢ`, `∂y/∂xⱼ = pⱼ(vⱼ − y)`. Contracting with an
upstream `dO` over the value channels gives FlashAttention-2's
`dSⱼ = Pⱼ(dPⱼ − D)` with `dP = dO·vᵀ` and `D = Σ dO∘O`, the gradient tang
derives for the online-softmax monoid.
*Proof.* Quotient rule on `N/D` with `N' = e^{xⱼ}vⱼ`, `D' = e^{xⱼ}`. ∎

## Part II — IO lower bounds

**Model.** Red–blue pebbling (Hong & Kung 1981): a two-level memory, fast
memory of `M` words, unbounded slow memory; I/O = words moved. Bounds are
over all schedules of the comprehension graph (the CDAG), with or without
recomputation. On a GPU we take `M` = shared memory one workgroup can
stage (3090: 48 KB/block = 12288 f32 words; M4: 32 KB = 8192), since
workgroups do not share staged data; registers and L2 are not credited.

**Theorem 11 (compulsory).** Every live graph input must be read once and
every output written once. An input read only through gather views
contributes at most one slice per index point.

**Theorem 12 (HBL bound per comprehension; `TensorGraph::io_hbl`).** Let a
reduction iterate a box `V` and read graph inputs through pick views with
coordinate sets `φⱼ` (plus its output's point coordinates if it is a
graph output). For any `s ∈ [0,1]^J` with `Σ_{j: q∈φⱼ} sⱼ ≥ 1` for every
coordinate `q`, `Q ≥ M·⌊|V| / F⌋` with `F = ∏ⱼ (2M sⱼ/σ)^{sⱼ}`,
`σ = Σ sⱼ`.
*Proof sketch.* Cut the schedule into stretches of `M` I/Os. In a stretch
at most `2M` words are available in total (`M` resident + `M` moved). By
the discrete Hölder–Brascamp–Lieb inequality (Christ et al. 2013;
Loomis–Whitney for matmul, Irony–Toledo–Tiskin 2004) the body
evaluations with all operands available number at most `∏ |φⱼ(E)|^{sⱼ}`,
maximized under `Σ|φⱼ(E)| ≤ 2M` at `|φⱼ(E)| ∝ sⱼ`. Intermediates are
excluded because a schedule may recompute them in fast memory. Any
feasible `s` gives a valid bound; tang searches a grid of twelfths. ∎
For matmul it gives `≈ 1.84·mnk/√M`, below COSMA's `2mnk/√M + mn`.

**Theorem 13 (matmul; cited).** Kwasniewski et al. (SC'19, Thm 1): classical
`m×k·k×n` matmul needs `≥ 2mnk/√M + mn` I/Os.

**Theorem 14 (attention forward; cited, constant ours).** Saha & Ye
(ICML'24): `Θ(N²d²/M)` for `M ≥ d²` (FlashAttention is optimal there) and
`Θ(N²d/√M)` for `M < d²`; constants are not stated. Their counting lemma
(one stretch produces at most `2M²/d²` entries of `QKᵀ`) gives
`Q ≥ S·d²/(2M) − M` with `S` the number of scores (`N²`, or `N(N+1)/2`
causal); for `M < d²` we use Theorem 12 on `(i, j, c)` with `s = (½,½,½)`.
The constant is derived from the lemma, not from a stated theorem.

**Theorem 15 (attention backward; derived, not mechanized).** Every score
must be produced again (recomputed from `Q, K` or read), so the stretch
counting of Theorem 14 applies with `dP = dO·Vᵀ` sharing the budget:
`Q ≥ S·d²/(3M) − M`, plus the compulsory `8Nd + N`. FlashAttention (§3.2)
claims `Θ(N²d²/M)` for its backward; no published lower bound covers it.

**Theorem 16 (row normalizers: softmax, RMSNorm, LayerNorm).** Per row of
`C`, `Q ≥ 2C + max(0, C − M)`.
*Proof.* No output of the row can be written before all `C` inputs have
been read (each output depends on the normalizer, which depends on every
input). At the moment of the first output write at most `M` inputs are
resident; each output needs its own input again, so at least `C − M`
inputs are read twice. Add the `C` writes. ∎

**MoE / scans.** The gathered expert GEMM is bounded by the compulsory
touched slices and, assuming balanced routing, Theorem 13 per expert. A
chunked linear-attention / gated-delta scan with state `dk×dv ≤ M` is
bounded (tightly) by its compulsory traffic; the closed form is
`tensor::io::bounds::chunked_scan`. Scans are not in the table yet: the
scan kernels are being reworked in a sibling branch.

## The IO table

`bound` = best lower bound (cited closed form ∨ tang's generic bound);
`tang generic` = Theorems 11–12 from the IR alone; `best known` = a
FlashAttention-2 schedule with `B = ⌊M/4d⌋` (an *upper* bound on the
optimum, `tensor::io::achievable`); `staged` = traffic of the schedule
tang picks (`TensorGraph::io_report`), each workgroup loading what its
tile needs, no credit for cross-workgroup cache hits; `once` = each kernel
reading each buffer once, which is `memory_axis`'s `hbm_bytes` (✓ = equal
to the value recomputed from the lowered WGSL program). Attention uses
the FlashAttention-2 placement (`MemorySchedule::flash`) and causal masks.
MB at 4 bytes/word.

| kernel | shape | device (M words) | bound MB | tang generic MB | best known MB | staged MB | once MB (wgsl check) | kernels | staged/bound | bound source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| matmul | 4096×4096·4096×4096 | 3090 (12288) | 5026.51 | 4555.46 | – | 8657.04 | 201.33 | 1 | 1.72 | Kwasniewski'19 Thm 1 |
| matmul | 512×576·576×1536 | 3090 (12288) | 35.84 | 29.98 | – | 59.77 | 7.86 | 1 | 1.67 | Kwasniewski'19 Thm 1 |
| matmul | 128×1536·1536×576 | 3090 (12288) | 8.47 | 7.47 | – | 14.45 | 4.62 | 1 | 1.71 | Kwasniewski'19 Thm 1 |
| attention fwd (flash) | N=1024 d=64 causal | 3090 (12288) | 1.05 | 1.05 | 6.75 | 34.08 | 1.05 ✓ | 1 | 32.50 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=1024 d=64 causal | 3090 (12288) | 2.10 | 1.84 | 20.33 | 184.04 | 10.77 ✓ | 8 | 87.59 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=4096 d=64 causal | 3090 (12288) | 5.54 | 4.19 | 94.05 | 538.97 | 4.19 ✓ | 1 | 97.21 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=4096 d=64 causal | 3090 (12288) | 8.40 | 7.34 | 283.56 | 2249.26 | 43.07 ✓ | 8 | 267.61 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=4096 d=128 causal | 3090 (12288) | 71.15 | 8.39 | 365.63 | 2151.68 | 8.39 ✓ | 1 | 30.24 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=4096 d=128 causal | 3090 (12288) | 71.15 | 14.68 | 1099.67 | 8795.52 | 86.07 ✓ | 8 | 123.62 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=16384 d=64 causal | 3090 (12288) | 89.43 | 16.78 | 1449.91 | 8598.32 | 16.78 ✓ | 1 | 96.14 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=16384 d=64 causal | 3090 (12288) | 89.43 | 29.36 | 4372.12 | 33206.57 | 172.29 ✓ | 8 | 371.29 | Saha–Ye'24 (const. from Lemma) |
| softmax | 4096×4096 | 3090 (12288) | 134.22 | 134.22 | – | 201.88 | 201.39 | 2 | 1.50 | row argument (docs/proofs.md) |
| softmax | 4096×32768 | 3090 (12288) | 1409.29 | 1073.74 | – | 1614.84 | 1610.68 | 2 | 1.15 | row argument (docs/proofs.md) |
| rmsnorm | 4096×4096 | 3090 (12288) | 134.22 | 134.22 | – | 201.61 | 201.36 | 2 | 1.50 | row argument (docs/proofs.md) |
| rmsnorm | 4096×32768 | 3090 (12288) | 1409.29 | 1073.74 | – | 1612.73 | 1610.65 | 2 | 1.14 | row argument (docs/proofs.md) |
| MoE expert GEMM (gather) | T=8 k=10 E=128 2048→512 | 3090 (12288) | 336.36 | 335.77 | – | 337.02 | 537.10 | 1 | 1.00 | compulsory + Kwasniewski per expert |
| MoE expert GEMM (gather) | T=4096 k=8 E=64 2048→1408 | 3090 (12288) | 7003.73 | 956.43 | – | 379753.07 | 956.43 | 1 | 54.22 | compulsory + Kwasniewski per expert |
| matmul | 4096×4096·4096×4096 | m4 (8192) | 6141.11 | 5579.31 | – | 8657.04 | 201.33 | 1 | 1.41 | Kwasniewski'19 Thm 1 |
| matmul | 512×576·576×1536 | m4 (8192) | 43.18 | 36.77 | – | 59.77 | 7.86 | 1 | 1.38 | Kwasniewski'19 Thm 1 |
| matmul | 128×1536·1536×576 | m4 (8192) | 10.30 | 9.18 | – | 14.45 | 4.62 | 1 | 1.40 | Kwasniewski'19 Thm 1 |
| attention fwd (flash) | N=1024 d=64 causal | m4 (8192) | 1.05 | 1.05 | 9.18 | 34.08 | 1.05 ✓ | 1 | 32.50 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=1024 d=64 causal | m4 (8192) | 2.10 | 1.84 | 27.66 | 184.04 | 10.77 ✓ | 8 | 87.59 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=4096 d=64 causal | m4 (8192) | 8.36 | 4.19 | 137.38 | 538.97 | 4.19 ✓ | 1 | 64.49 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=4096 d=64 causal | m4 (8192) | 8.40 | 7.34 | 414.22 | 2249.26 | 43.07 ✓ | 8 | 267.61 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=4096 d=128 causal | m4 (8192) | 87.17 | 8.39 | 543.18 | 2151.68 | 8.39 ✓ | 1 | 24.68 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=4096 d=128 causal | m4 (8192) | 87.17 | 14.68 | 1633.71 | 8795.52 | 86.07 ✓ | 8 | 100.91 | Saha–Ye'24 (const. from Lemma) |
| attention fwd (flash) | N=16384 d=64 causal | m4 (8192) | 134.19 | 16.78 | 2160.13 | 8598.32 | 16.78 ✓ | 1 | 64.07 | Saha–Ye'24 (const. from Lemma) |
| attention fwd+bwd (flash) | N=16384 d=64 causal | m4 (8192) | 134.19 | 29.36 | 6513.89 | 33206.57 | 172.29 ✓ | 8 | 247.45 | Saha–Ye'24 (const. from Lemma) |
| softmax | 4096×4096 | m4 (8192) | 134.22 | 134.22 | – | 201.88 | 201.39 | 2 | 1.50 | row argument (docs/proofs.md) |
| softmax | 4096×32768 | m4 (8192) | 1476.40 | 1073.74 | – | 1614.84 | 1610.68 | 2 | 1.09 | row argument (docs/proofs.md) |
| rmsnorm | 4096×4096 | m4 (8192) | 134.22 | 134.22 | – | 201.61 | 201.36 | 2 | 1.50 | row argument (docs/proofs.md) |
| rmsnorm | 4096×32768 | m4 (8192) | 1476.40 | 1073.74 | – | 1612.73 | 1610.65 | 2 | 1.09 | row argument (docs/proofs.md) |
| MoE expert GEMM (gather) | T=8 k=10 E=128 2048→512 | m4 (8192) | 336.36 | 335.77 | – | 337.02 | 537.10 | 1 | 1.00 | compulsory + Kwasniewski per expert |
| MoE expert GEMM (gather) | T=4096 k=8 E=64 2048→1408 | m4 (8192) | 8536.30 | 956.43 | – | 379753.07 | 956.43 | 1 | 44.49 | compulsory + Kwasniewski per expert |

### Reading it

* **Matmul is within 1.4–1.7× of the COSMA bound.** tang's 16×16 tile
  with a 4×4 register tile stages a 64×64 output block; the bound's
  `√M` is 111 (3090) / 91 (M4). A 96×96 or 112×112 block — what the
  bound says fits — would close most of the gap. Since `2mnk/√M` is
  achievable up to `√S/(√(S+1)−1)` (Kwasniewski), the ratio is the
  schedule's slack, not the bound's.
* **Row normalizers are within 1.1–1.5×.** tang runs statistics and
  normalize as two kernels, reading `x` twice. When a row fits in fast
  memory (4096 < M) the bound is one read and one write; a single kernel
  keeping the row staged would meet it. For rows longer than `M` the
  two-pass schedule is within 1.15× of optimal.
* **Attention: the flash placement is asymptotically the right one, its
  tile is not.** The scalar WGSL flash kernel stages 16 query rows per
  workgroup, so K and V are re-read `N/16` times (and the model does not
  credit skipped masked blocks). The best known schedule uses
  `B = M/(4d)` (48 rows on a 3090 at d = 64) and is 3.7–5.9× lower than tang's
  staged traffic; the remaining 5–17× to the lower bound is the
  bound's unknown constant (Theorem 14 is tight only up to it). For
  fwd+bwd (8 kernels) tang's staged traffic is 6.5–9× FA-2's.
* **MoE.** The token-major gather schedule (decode) meets the compulsory
  bound at T = 8. At prefill size the same schedule re-reads each
  expert's slice per (token, slot) — 44–54× the bound — which is why the
  prefill path sorts tokens by expert and reduces over segments (PR #168);
  that path is not modelled here.

## Validation

* `once` reproduces `memory_axis`'s `hbm_bytes` exactly for every
  attention row (fwd and fwd+bwd, N = 1024…16384, d = 64/128), which
  matches `paper/tang/data/memory-axis.json`'s flash rows byte for byte
  (e.g. 10 768 384 B for fwd+bwd at N = 1024, 172 294 144 B at N = 16384).
  This checks that the analysis sees the same kernels and buffers as the
  lowering. It is a model-to-model check: `hbm_bytes` is itself a model.
* Staged traffic ≥ every lower bound in every row (unit test
  `matmul_bound_below_cosma_and_staged_traffic_above_it`, and the table).
* Not yet done: DRAM counters (`ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum`
  on the 3090, Metal `GPU counters` on the M4). Staged traffic should
  sit at or above the measured bytes (caches can only help); the gap is
  how much L2 rescues the small tiles.

## What is not proven

* **Floating point.** All theorems are over ℝ. Reassociation (Theorems
  3, 5), rescaling by `e^{m−M}` (Theorem 6) and shifting (Theorem 7)
  change f32 rounding; the online state cannot overflow where two-pass
  could not, but no error bound is mechanized.
* **The Rust is not extracted from the Lean.** The Lean model mirrors
  `Form::compose`, the online-softmax combine and the VJP rule; that the
  Rust implements exactly these is checked by tang's tests against the
  f64 interpreter, not proved.
* **Guarded and gathered views in fusion.** Theorem 2 covers affine
  views. Guards (zero padding) and gathers compose in the Rust but their
  composition lemmas are not mechanized; `substitute_maps` already
  refuses to splice a map through a guard (since `f(0) ≠ 0` in general).
* **Other gradients.** Theorem 8 is for sum comprehensions; gradients of
  max/argmax/top-k reductions (subgradients at ties), scans and operator
  nodes are not covered beyond Theorems 9–10.
* **Other rewrites.** `schedule_inline`, `merge_siblings`,
  `hoist_invariant_factors`, `merge_reduce_dims`, relayouts and
  precision schedules are not mechanized (they are schedule decisions or
  algebra outside the four core theorems).
* **IO bounds are not mechanized.** HBL / Loomis–Whitney, the cited
  matmul and attention bounds, and Theorems 15–16 are paper proofs. The
  attention constants in Theorems 14–15 are derived from a lemma's
  statement, not from a stated theorem, and are likely loose.
* **The staged-traffic model** ignores caches across workgroups, skipped
  masked tiles, and asynchronous copies; it is an exact count of the
  loads the scalar kernels issue only for the WGSL/MSL lowering.

## References

Hong & Kung, *I/O complexity: the red-blue pebble game*, STOC 1981.
Irony, Toledo, Tiskin, *Communication lower bounds for distributed-memory matrix multiplication*, JPDC 2004.
Ballard, Demmel, Holtz, Schwartz, *Minimizing communication in numerical linear algebra*, SIMAX 2011.
Christ, Demmel, Knight, Scanlon, Yelick, *Communication lower bounds and optimal algorithms for programs that reference arrays*, 2013.
Kwasniewski et al., *Red-blue pebbling revisited: near optimal parallel matrix-matrix multiplication*, SC 2019.
Dao et al., *FlashAttention*, NeurIPS 2022.
Saha & Ye, *The I/O complexity of attention, or how optimal is FlashAttention?*, ICML 2024.
