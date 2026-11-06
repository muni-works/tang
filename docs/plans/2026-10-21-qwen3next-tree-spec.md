# Tree speculative decoding for Qwen3-Coder-Next

Goal: single-stream decode from ~246 tok/s toward 500+ on one H200, output identical to greedy.
Builds on #113 (sequence slots: per-row `(pos, slot)` for attention, per-segment GDN).

## Why a tree

A verify window of T rows costs c(T) T=1 steps; measured c(2) ≈ 1.35, c(8) ≈ 3.3
(≈ 1 + 0.33 (T − 1)). One chain of k drafts with per-token acceptance a_i yields
E = 1 + a_1 + a_1 a_2 + … tokens. On fresh text the suffix drafter's first token is right maybe
30–50% of the time, so the long tail of a chain is almost never reached: the rows spent on draft
4..7 are wasted. Spending those rows on a *second candidate for the first few positions* raises
P(first draft accepted) from p₁ to p₁ + p₂ (disjoint candidates), which is where the expected
tokens are.

## Tree shape: a forest of chains sharing the root

Window rows (≤ MAX_T = 8): chain b is `[cur, d_b1, d_b2, …]`, contiguous rows `r0_b .. r0_b + L_b`.
Every chain repeats `cur` at position pos0. No deeper sharing.

Why not a general tree: GDN is a recurrence. A general tree needs the state after an interior
node as the start of each subtree, i.e. a GDN kernel walking row-index lists with forked
states. A forest of chains is exactly #113's segment model (segment = chain, all segments read
slot 0's state), so the recurrence needs **no kernel change**. The cost is one duplicated root
row per extra chain, and duplicated rows route to the same experts, so with grouped experts
(#113 M2) they add attention/GDN/dense cost but almost no expert bandwidth — the dominant term.

Shapes we'll try (rows): `[8]` (today), `[5,3]`, `[4,4]`, `[4,2,2]`, `[3,3,2]`. A shape is picked
per step by the same expected-tokens-per-ms rule `decode_spec_options` uses, over the measured
window cost EMA.

Draft sources for chains:
1. suffix automaton: the best continuation (today's draft) plus the 2nd/3rd most frequent
   children of the matched state (`Sam::children`), each extended greedily;
2. the MTP head (Coder session) when it lands: chain 0 = MTP top-1 path, chain 1 = MTP top-2 at
   depth 1 then MTP greedy, or the suffix chain when it disagrees with MTP;
3. the logits of the previous window give no new information for row 0 (that is `cur`), so they
   don't seed branches.

Distinct chains must differ at depth 1 (dedupe by first token), else they're wasted rows.

## Verify

One window, `verify = true`, rows `(pos0 + depth, slot 0)`, segments `[r0_b, L_b, 0, 0]`.

- **GDN**: in verify mode the state is only read, so every segment reading slot 0's state
  (`Stab[0]`, `Htab[0]`) is correct and race-free. Commit: re-stage segments with
  `n_keep = 0` for every chain except the winner (`n_keep = m + 1`), run the per-segment commit
  (`fl_gdn_conv_step_seg` with commit, `fl_gdn_conv_commit_seg`). The winner writes slot 0's state
  in place — no state copy (state is 75 MB; copying it per branch would cost ~40 µs each).
- **Attention** — the one kernel change. Rows of different chains at the same depth have the same
  position, so with the cache indexed by position they'd overwrite each other's K/V. Add one
  field per row, `off` (cache offset for window keys): chain b's non-root rows write K/V at
  `pos + off_b` with `off_b = Σ_{c<b} (L_c − 1)`, and a row at chain b reading key position
  `p > pos0` reads cache index `p + off_b`. Rope still uses `pos`. Keys `≤ pos0` (context and the
  shared root) are unchanged. Chain 0 has `off = 0`, so today's single-chain windows are
  bitwise unchanged.
  Touches `cx_attn_prep` / `cx_attn_part` (tang-pro's `coder_kernels.rs`): one more `u32` per row in
  the staged `rows` record (`rows[2 MAXT + t]`).
- **Accept**: per chain, m_b = longest prefix with `d_bi == argmax(row i−1)`; the winner is the
  max m_b (ties → lower b). Emit `argmax` of the winner's rows `0..m`. If `off_winner ≠ 0`, move
  its m K/V entries from `pos0 + 1 + off` to `pos0 + 1` for the 12 attention layers (one small
  D2D copy per layer and K/V, captured in a graph per (m, off)).

Correctness: every emitted token is the argmax of a row whose whole causal input (context, root,
the chain's accepted drafts) equals the greedy sequence's, and #113 shows rows are bitwise
independent of what else shares the window. So output == greedy bitwise. Sampling: same
argument per row with the row's sampler key = (pos, counter) as today; the accept rule becomes
"draft equals the row's sample", which is the standard lossless rule for a deterministic sampler.

## Cost model / what to expect

Per step: E(shape) / c(rows). With c(T) = 1 + 0.33 (T − 1) and suffix acceptance on fresh text
(p₁ ≈ 0.35, p₂ ≈ 0.15, later tokens ≈ 0.6 conditional):

| shape | rows | c | E | E/c |
|---|---|---|---|---|
| none | 1 | 1.00 | 1.00 | 1.00 |
| [8] | 8 | 3.31 | 1.83 | 0.55 |
| [4] | 4 | 1.99 | 1.69 | 0.85 |
| [2] | 2 | 1.33 | 1.35 | 1.02 |
| [3,2] | 5 | 2.32 | 1.71 | 0.74 |

So on fresh text the tree **does not pay at today's c(T)**: c(T) is too steep. Trees pay when
(a) drafts are good (refactors: already ~400 tok/s with one chain; the second chain adds the
cases where the file diverges from the earlier copy), and (b) c(T) flattens — with grouped
experts plus a megakernel the marginal row is attention+dense only. At c(T) = 1 + 0.12 (T − 1)
(dense-only marginal cost) and an MTP head with p₁ ≈ 0.75: [4] gives E ≈ 2.9 at c = 1.36 →
2.1× → ~500 tok/s; [4,2] adds ≈ +0.15 tokens at +0.24 c — marginal.

Conclusion for planning: **500 tok/s comes from MTP acceptance × flat window cost; tree is a
+5–15% multiplier on top, not the main lever.** Build it cheaply (forest of chains, one
attention field), gate it by the measured cost table, and measure.

## Plan

1. `flash/tree.rs`: forest builder (suffix + optional MTP), per-chain acceptance, shape choice;
   pure host code, unit-tested.
2. Kernel hook (ask tang-pro): `off` row field in `cx_attn_prep` / `cx_attn_part`, K/V compaction
   copy. `Coder::verify_forest(cur, &[Vec<u32>])` on #113's `stage_rows`.
3. Test: `flash-tree-check` on the tiny qwen3next GGUF (`scripts/qwen3next_tiny.py`) on mew's
   3090: decode N tokens greedy and with forced forests (one right chain among wrong ones, right
   chain in each position), tokens and logits bitwise equal.
4. H200 n=3 on `docs/bench/h2h` fresh-context harness: none / suffix chain / suffix forest.
