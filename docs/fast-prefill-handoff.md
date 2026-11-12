# Fast prefill (#280): the 4-way bisect and the 32k NaN

Status: **GPU frozen**. Everything below that needs a GPU is written as an
exact `scripts/gpu-run.sh` command and has **not** been run. The CPU-answerable
parts have been run; their results are in this doc.

Branch: `claude/moe-align` at `9747c13f`, PR muni-works/tang#280.
Scoring: teacher-forced with `coder_derived --check 33` (223 positions).

---

## 1. The four changes, as committed

The PR is a stack of independent commits, which is what makes the bisect
possible at all:

| # | commit | change | bisectable how |
|---|---|---|---|
| 1 | `e254be6d` | `ScanKind::Sort`: a counting sort (one CUDA block) replaces the O(P²) rank and picks in `schedule_expert_groups` | code revert only |
| 2 | `1eb3794e` + `db3421b8` | `substitute_maps`: a map stays materialized when its reshape feeds a scan/operator, or is read through a gather | code revert only |
| 3 | `1e4dfbc8` | `coder_derived`: grouping on by default (`--group 16`) | **runtime**: `--group 0` |
| 4 | `e80be5f0` | the Q6_K routed down-projection goes up as Q8_0 and runs the int8 kernels | **runtime**: `TANG_Q6K_Q8=0` (added in this branch) |
| + | `a3bc106f` | MMA tile refuses gathered/guarded operands (cherry-picked from ds-prefill `8cac0be3`) | code revert only |
| + | `e53de284` | `split_bilinear` factor cap 2³¹ (`TANG_SPLIT_MAX`) | env only |

Two of the four are runtime-switchable. **Two are not**, and one of those is
structurally inseparable from another — see §3.

### Why `TANG_Q6K_Q8` had to be added

`coder_derived` hardcoded `q6k_as_q8_0: true`, so arm 4 could not be scored
against arm 3 without a code change. `coder_batched` already reads
`TANG_Q6K_Q8`; `coder_derived` now does too (default on, so nothing changes
for existing callers). That is the only behavioural edit besides the clippy fix.

---

## 2. What the CPU path already answers

`cargo test -p tang-bench --lib cuda_cpu -- --nocapture` — **11/11 pass**.
The grouping machinery is numerically correct against the interpreter on the
tiny coder (6 experts top-2, d=32):

```
layer-wise k = 16 (group 0): [107, 105] kernels, logits 1.14e-6 (chunked 1.14e-6), caches vs chunked 0.00e0
layer-wise k = 16 (group 2): [155, 153] kernels, logits 1.14e-6 (chunked 1.14e-6), caches vs chunked 0.00e0
```

Grouped and ungrouped agree to the same 1.14e-6 and produce **bit-identical
caches** vs chunked prefill. The sort is a permutation, so the sums are the
same terms in the same order — as the module docs claim.

**This rules out the counting sort and `substitute_maps` as accuracy
regressions.** Both are exercised here and both are exact. It does *not* clear
them for the NaN at 32k — see §4.

The CPU path cannot answer the 194/198-vs-200/206 question: the tiny model has
6 experts and 32-wide rows, so no real quantization or routing geometry is
present.

---

## 3. The bisect can't be a clean 4-way — say so plainly

The intended arms were:

```
sort only
sort + substitute_maps
+ grouping
+ Q8_0 requant
```

**"Sort only" does not exist.** Reverting both `substitute_maps` commits
(`1eb3794e`, `db3421b8`) and keeping the sort makes grouping **panic**:

```
thread '...layerwise_prefill_matches_chunked_on_the_cpu' panicked at
crates/tang-compile/src/tensor.rs:1942:56: called `Option::unwrap()` on a `None` value
  ... TensorGraph::substitute_maps_with
  ... cuda_cpu::tests::chunked_then_decode_grouped
```

(line 1942 is the reverted tree; the site is the `remap[t.n()].unwrap()` in
`substitute_maps`' output remap, reached from the grouped pass.)

The sort is fed `ng.reshape(idx, …)` — a **reshape of a map**. That is exactly
what `1eb3794e` protects. Without it the map is spliced away, the reshape's
source is dropped, and `remap[src.n()].unwrap()` panics. The same is true of
the `at_rank` picks, which read the reshape through a gather (`db3421b8`).

So the honest decomposition is **three** arms, not four:

| arm | what changes | vs previous |
|---|---|---|
| **A** | `--group 0` `TANG_Q6K_Q8=0` | the PR's ungrouped baseline |
| **B** | `--group 16` `TANG_Q6K_Q8=0` | + sort + substitute_maps + grouping |
| **C** | `--group 16` `TANG_Q6K_Q8=1` | + Q8_0 requant |

and separately, on any of them:

| arm | what changes |
|---|---|
| **D** | revert `a3bc106f` (MMA gather refusal) |

`substitute_maps` is a *prerequisite* of the sort, not an independent variable.
It cannot cost accuracy — it only prevents a graph rewrite; every CPU test
above runs through it and is exact.

### The gather refusal is a correctness fix, and it applies to CUDA too

`a3bc106f` lands in `tensor_msl_mma.rs`, but `bilinear_tile` is the analysis
**shared** by the CUDA tile (`tensor_cuda.rs:446`), OpenCL
(`tensor_opencl_gemv.rs:1355`), fp8 (`tensor_cuda_fp8.rs:88`) and attn_bwd
(`tensor_cuda_attn_bwd.rs:632`). The CUDA tile indexes operands affinely
through `msl_mma::Addr` and has no gather handling, so before the fix a
gathered operand on the CUDA tile read the wrong elements. The commit message
frames it as DS sparse attention since #234; the blast radius is wider.

Two consequences for this PR:

1. **Arm D should not be "does it move the score" — it should be "is it
   correct".** Reverting it to regain a few points on `coder_derived` would be
   reintroducing a known wrong-rows bug. Score it, but don't trade on it.
2. The grouped contraction in `group_one` reads **both** the weight (gathered)
   and the activation (gathered), so grouping depends on the int8 path's own
   gather support (`tensor_cuda_int8.rs:268-292`), *not* on the bilinear tile.
   That is the right outcome, but it means the arm-C score also exercises the
   int8 gathered-MMA path — the one place a real-model-only error could hide.

---

## 4. The 32k NaN

Panic: `coder_derived.rs:253`, `partial_cmp(..).unwrap()` returning `None`
because a logit is NaN.

### The PR's explanation is the wrong quantity

The PR body blames `TANG_SPLIT_MAX` at rows·16·ctx = 4096·16·33024 ≈ 2.16e9.

That number is **the wrong quantity**. The cap is checked **per comp**
(`tensor_cuda_split.rs:241`, inside the `for r in …` loop), and each head is
its own comp. At 32k with chunk 4096:

```
per-head factor = 4096 × 33024 = 1.353e8 elements = 0.54 GB
cap default 2³¹ = 2.147e9  ->  per-head fits by 16×
cap 2²⁹ (the old MAX_FACTOR) = 5.369e8  ->  per-head still fits
```

The 4.4 GB figure in the code's own comment is **all 16 heads summed**, which
is not what the cap compares. So at 32k **the split is not refused** — p·v is
already a tensor-core matmul, and `TANG_SPLIT_MAX` never enters the decision.
Raising or lowering it changes nothing at this geometry.

The manager's read is right, and it's stronger than "slow, not NaN": past the
cap p·v falls back to a warp reduction (slow, and correct — the tiny-model
`split_bilinear_keeps_the_step` test runs both paths at rel err 0.00e0), and at
this geometry the cap isn't even hit.

### The counting sort is int-safe at this size

Checked directly. At 32k with top-10, P = 327,680; `np` is a compile-time
literal in the kernel, `int` reaches 2.1e9, and `out[np + r]` peaks at
2·327,680 ≈ 6.6e5. `experts` × 4 bytes of shared memory is a few KB. No int32
overflow, no shared-memory overflow. **Not the NaN.**

### Ranked suspects, in order

1. **Attention, at long context.** Unchanged in this PR, which is the point:
   if it reproduces on unmodified `origin/main`, the PR did not cause it.
   This is the branch the manager flagged — if it traces here, flash-attn #275
   comes back as a correctness fix, not a speed one.
2. **`substitute_maps` at 32k rows.** It is exact at k=16 on the tiny model
   (§2) but the 32k step is a different size regime. Cheap to isolate by code
   revert.
3. **The grouped path at 32k rows.** `--group 0` isolates it with no rebuild.
4. **The split cap.** Least likely, and arithmetic says it can't be reached at
   this geometry.

The discriminating control is **`origin/main` at 32k** — no grouping, no sort,
no Q8_0, cap at the old 2²⁹. If main NaNs too, suspects 2 and 3 are out
immediately and this is a pre-existing long-context bug.

---

## 5. Staged GPU commands (NOT RUN — GPU frozen)

`scripts/gpu-run.sh <pod> <ref> cmd:<shell> [KEY=VAL …]`, `LANE=short`.
Pod: `h200-1` for prefills. Model: `/models/coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`
(the runner copies it to `~/runner/models/coder-next` and exports `MODEL_DIR`).

The example builds with `--features cuda,gguf`; the runner's `cmd:` job does
not build, so each command builds first.

### Score the three arms (the attribution table)

```bash
# A — ungrouped baseline (no sort, no grouping, Q6_K stays f32)
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids --ctx 4096 --chunk 128 --check 33 --group 0 --top 5" \
  TANG_Q6K_Q8=0

# B — + sort + substitute_maps + grouping
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids --ctx 4096 --chunk 128 --check 33 --group 16 --top 5" \
  TANG_Q6K_Q8=0

# C — + Q8_0 requant
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids --ctx 4096 --chunk 128 --check 33 --group 16 --top 5" \
  TANG_Q6K_Q8=1
```

Repeat all three at `--chunk 256`. `/tmp/ids` is the 33-token ids file used for
the 223-position score — reuse the one behind the PR's 194/198/200/206 numbers
so the arms are comparable.

### Arm D — does the gather refusal move the score

Needs a branch with `a3bc106f` reverted. Build it, push, then re-run arm B
against it. Score it, but see §3: a lower score on the reverted branch is the
bug returning, not a win.

### The NaN: control first

```bash
# 1. the control — does UNMODIFIED main NaN at 32k?
LANE=long scripts/gpu-run.sh h200-1 origin/main \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids32k --ctx 33024 --chunk 4096 --check 33 --group 0 --top 5"

# 2. PR head, grouping off (isolates the sort + substitute_maps + split cap)
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids32k --ctx 33024 --chunk 4096 --check 33 --group 0 --top 5" \
  TANG_Q6K_Q8=0

# 3. PR head, grouping on, cap lowered as the PR suggests
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids32k --ctx 33024 --chunk 4096 --check 33 --group 16 --top 5" \
  TANG_Q6K_Q8=1 TANG_SPLIT_MAX=536870912

# 4. grouping on, Q8_0 off (does the requant produce the NaN?)
LANE=long scripts/gpu-run.sh h200-1 claude/moe-align \
  "cmd:cargo build --release -p tang-bench --features cuda,gguf --example coder_derived && \
   target/release/examples/coder_derived \$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
     --ids-file /tmp/ids32k --ctx 33024 --chunk 4096 --check 33 --group 16 --top 5" \
  TANG_Q6K_Q8=0
```

Read: 1 NaN ⇒ pre-existing long-context bug, PR did not cause it (go to
attention, revive #275). 1 clean but 2 NaN ⇒ sort/`substitute_maps`/cap.
2 clean but 3 NaN ⇒ grouping. 3 clean but 4 NaN ⇒ the Q8_0 requant.

`TANG_SPLIT_MAX=536870912` in command 3 is included only because the PR asks
for it; per §4 it cannot change the outcome at this geometry, so treat a
difference there as evidence the model of the cap is wrong, not as a fix.

---

## 6. Merge recommendation

**Hold the bundle as-is; land it without Q8_0 (arm B).**

- Clippy is clean at the CI toolchain (§7) — the PR's only red check.
- The sort and `substitute_maps` are **not** accuracy regressions: exact on the
  CPU, and `substitute_maps` is a hard prerequisite of the sort (§3).
- Grouping is the only remaining accuracy variable, and the CPU path shows it
  is an exact reordering, so the 194-vs-200 gap most likely comes from arm C
  (the Q8_0 requant of the Q6_K down-projection — the one arm that changes
  numbers rather than order). **That is a hypothesis, not a result**: it needs
  the three-arm GPU score above to confirm.
- Do not revert `a3bc106f` to buy accuracy; it is a correctness fix that
  reaches CUDA (§3).
- The 32k NaN should not gate the merge if command 1 shows main NaN-ing too.

`~1.3×` over main without Q8_0 is the floor worth landing.

---

## 7. Clippy

`crates/tang-compile/src/tensor_scan.rs:514` cloned a `Dim`, which is `Copy`
(`tensor.rs:946-947`). CI runs `cargo clippy … -- -D warnings`, so
`clippy::clone_on_copy` failed the job. `.clone()` removed.

Verified with the **CI toolchain**, not the local default (local stable is
1.98.1; CI pins 1.94.1, and the newer clippy fires lints that don't exist in
1.94 — `chunks_exact_to_as_chunks` in `tang-gpu`, a collapsible `if` in
`tang-3dgs`, neither file touched by this PR):

```
cargo +1.94.1 clippy --workspace --lib --tests --bins \
  --exclude tang-site-wasm --exclude tang-sheffer-wasm -- \
  -D warnings -A clippy::too_many_arguments -A clippy::type_complexity \
  -A clippy::needless_range_loop -A clippy::ptr_arg
  -> Finished `dev` profile, exit 0
```

---

## 8. Open items

- The three-arm score (§5) is unrun. Everything in §6 past "sort and
  substitute_maps are clean" is inference from the code, not measurement.
- Arm D (gather refusal) needs a revert branch.
- The 32k NaN verdict needs command 1 of §5 — the control is the whole answer.
- `--ctx 33024` assumes a 32k prompt plus one chunk of headroom; confirm the
  ids file's length matches what the PR used.
