# Coder-Next: SpecLA-style GDN verify/commit

Status: phase 1 (CPU) done on `frog/coder-specla`; GPU gates and numbers pending.
Paper: SpecLA, arXiv 2607.16673 (factor buffering, chain-decomposed tree verify, delayed
commit; measured on a 1.3B pure-GDN model, h_v=8, d=256, H100).

## How a verify window updates GDN state today

Geometry (Coder-Next): 36 GDN layers, 32 v heads × 16 k heads, d = 128. State 32×128×128 f32 =
2 MiB a layer (72 MiB total). Conv history 3 × 8192 channels. Stacked projection row
(`GdnL.proj`): qkv 8192 | z 4096 | a 32 | b 32 = 12,352 f32.

1. **Verify** (`enqueue_window`, `verify = true`): per layer, `gdn_in` writes all T rows of
   `proj`, then `fl_gdn_conv_step_seg` with commit 0: one block a v head loads the 2 MiB state
   into registers, runs conv + recurrence for all T rows, writes outputs, **never stores the
   state**. No snapshots, no restore.
2. **Commit** (`commit(t, n_keep)`, a CUDA graph per T): per GDN layer, `fl_gdn_conv_step`
   with commit 1 re-runs conv + recurrence over the first `n_keep` rows **from the buffered
   `proj`** and stores the state, then `fl_gdn_conv_commit` shifts the conv history. Tree
   windows (`commit_tree`) and batches (`commit_batch`) do the same per segment.

So the engine already does SpecLA's factor buffering: the buffer is `proj` (pre-conv q/k/v, z,
a, b per row, 395 KB a layer at T=8, which exists anyway). The commit never replays GEMVs, MoE
or attention, only the GDN recurrence (SpecLA's "DeltaConstruct"). The paper's 2.7–4.3×
commit win is replay-vs-factors; we are already on the factors side.

Commit cost: the stage sweep puts GDN conv+step at 238 µs a T=1 step for 36 layers (6.6 µs a
layer, latency-bound: serial per-token reductions, not the 2 MiB state read). The commit graph
is 72 small launches (replay of ~2.5 kept rows + history shift), estimated ~0.3–0.4 ms, about
3–4% of the ~9.6 ms T=8 window. To be measured on the GPU (`last_commit_ms`, stage sweep).

## Mapping SpecLA onto our kernels

| SpecLA | here |
|---|---|
| factor buffer (k, v, α, β per position) | `GdnL.proj` rows (qkv pre-conv, a → α via softplus/exp, b → β via sigmoid) plus the conv history |
| DeltaConstruct over the accepted prefix | `gdn_step_body` with commit, from `proj` |
| chain-decomposed tree verify, boundary states | a chain's segment runs its parent's prefix rows in its own registers (below) |
| delayed commit (pending factors applied in the next verify kernel) | not done yet (below) |

Extra memory from this work: none (the prefix rows already live in `proj`).

## What changed (phase 1)

1. **State-only commit replay** (`GDN_REPLAY` = commit | 2). The replay used to compute every
   kept row's output (second dot product, three block reductions, gated norm, store) only to
   throw it away. Now it does the state update alone, with the same fmas, so the state is
   bitwise the same as before. Roughly halves the per-token serial work of the commit kernel.
   Verify and plain windows (`GDN_VERIFY` / `GDN_COMMIT`) are unchanged.
2. **Chain-decomposed forest windows.** Before, each chain repeated the root `cur`, so a
   forest of k chains paid k−1 extra full rows (GEMVs, MoE, attention), about 0.8 ms each at
   T≈8. Now the root row is in the window once. A later chain's GDN segment is its own drafts
   with a **path prefix** (the root row): `SEG[3]` with bit 31 packs
   `keep | pr0 << 8 | plen << 16`, and `gdn_step_body` maps path index i to window row
   `i < plen ? pr0 + i : r0 + i − plen`, running the prefix rows without writing their
   outputs (chain 0's segment writes them). This is SpecLA's "serial" chain verifier with
   the boundary state recomputed in registers rather than stored to HBM: the prefix is
   one row, so recomputing it is cheaper than the round trip. Attention needed no change: the
   root's key at `pos0` is below every row's offset boundary (`CX_OFF`). The commit keeps the
   winner's path (root + m drafts) through the same packed segment and history shift. MTP
   refill gathers the winning path's hidden rows to rows 0.. first (`mtp_gather`).
   `tree::choose` now counts a later chain's cost without the root row.
   Branching deeper than the root (at depth d of chain 0) works in the GDN kernel
   (plen = d + 1) but needs a second cache-offset boundary in attention; not wired.

## Not done yet: delayed commit

Fold the commit into the next window's GDN step: carry `n_keep` rows of the previous window's
`proj` (ping-pong buffers, or a copy of the kept rows) and the pending count in `ctl`. The next
step kernel loads the state, applies the pending rows, then verifies its own rows, and stores
nothing extra. That saves the separate commit graph (72 launches, one state load/store each
layer). The cost is a second `proj` buffer, twice as many captured graphs (parity), and a
flush before anything that reads state directly (`save_running_at`, batch slots, prefill,
`step`). Worth it only if the GPU shows commit at 4% or more of the window after the
state-only replay. Decide in phase 2.

## Expected saving

- State-only replay: ~0.1 ms a window (~1%).
- Forest windows: one full row saved for every chain past the first (mtp-forest: 1 row,
  ~0.8 ms of a ~9 ms window, or one more draft of chain 0 in the same 8 rows). This is the
  change that can make tree verify pay. The GDN recurrence was never the bottleneck here:
  the duplicated rows were.
- Delayed commit (if built): ~0.2–0.3 ms a window more.

## Gates (phase 2)

`flash-spec-test -n 96` on `ids/explain.ids` (all modes SAME), `flash-tcheck` bitwise,
`flash-tree-check` (PASS, mtp and mtp-forest speed), `flash-aa -n 512 --bw 4800` on
explain/refactor/write in mtp and oracle modes before and after, T=8 window breakdown.
