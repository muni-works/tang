# FA2-style prefill attention (`TANG_CODER_ATTN=fa2`)

Wide prefill (#124) runs the 12 full-attention layers of Qwen3-Coder-Next with split-K
(`cx_attn_part` + `cx_attn_merge`): one block per (row, kv group, 64-key chunk), CUDA-core f32
math, each row re-reading every K/V cell. That's about 65% of prefill on an H200. The tiled
CUDA-core kernel in draft #128 (`cx_attn_tile`) shares K/V tiles across 8 rows but stays off the
tensor cores, and it came out 1-4% slower. This branch adds a FlashAttention-2-style kernel
(`cx_attn_fa2`) behind `TANG_CODER_ATTN=fa2`. Split-K stays the default.

The structure follows FlashAttention-2 (Tri Dao, arXiv 2307.08691;
github.com/Dao-AILab/flash-attention, BSD-3-Clause). No FA code was copied. The kernel is written
against this engine's layouts.

## Where it hooks in

- `crates/tang-llm/src/flash/coder_kernels.rs`: `cx_attn_fa2` (CUDA source in `SRC`). It is built
  into both the narrow module and the wide module, like every Coder kernel.
- `crates/tang-llm/src/flash/coder.rs`:
  - `Variants::attn_fa2` is read from `TANG_CODER_ATTN=fa2`, using the same env pattern as
    `TANG_CODER_WGEMM` / #128's `TANG_CODER_ATTN_TILE`.
  - `Kern::fa2` is set in both constructors. `set_smem` opts it into large dynamic shared memory,
    and it shows up as `attn_fa2` in `TANG_CODER_PROF`.
  - `enqueue_mixer` handles wide windows with fa2 on: `cx_attn_prep` (unchanged) runs first, then
    `cx_attn_fa2` over the whole window, then the existing `cx_attn_merge` over its key splits.
    `cx_attn_merge` also applies the sigmoid gate. `attn_out` follows.
  - `fa2_splits` picks the key-split count and falls back to split-K when the partials wouldn't
    fit.
- `crates/tang-llm/src/flash/fa2.rs` holds the launch planning (`FA2_BM`, `fa2_smem`,
  `fa2_splits`) and a CPU model test (see Tests).

Narrow windows (decode, verify, MTP) never take this path.

## Design

Inputs are the ones split-K reads. Q is f32 `[t][NH][HD]`, written by `cx_attn_prep` after
rmsnorm and rope. K and V sit in bf16 slot caches, `[pos][NKV][HD]`. Rows come from
`rows[]`: a wide window is one slot at consecutive positions. Qwen3-Next has NH=16, NKV=2 and
HD=256, so G=8 query heads per KV head. The kernel is generic over `CX_*`; it static-asserts
`64 % G == 0` and `HD % 16 == 0`.

- **GQA packing.** The M dimension of a tile is (row, head) pairs within a kv group, with
  vector `v = row*G + head`. With `Br = FA_BM = 64` vectors that's 8 rows × 8 heads. All 8 heads
  read the same K/V tile, so one K/V load feeds 64 query vectors. Split-K loads K/V once per
  row per group.
- **Tiles.** Br=64 vectors and Bc=32 keys, with 4 warps of 16 vectors each (block 128).
  - `S = Q·Kᵀ` is `mma.sync.m16n8k16` bf16→f32: 16 k-steps over HD and 4 n-tiles of 8 keys.
  - `O += P·V` reuses the S accumulators directly as the A fragments (the FA2 register trick).
    V's B fragments come from `ldmatrix.x4.trans`.
  - O is 16×256 f32 per warp, which is 128 registers a thread. Q lives in shared memory rather
    than registers so the kernel stays under 255 registers (FA2 makes the same choice at d=256).
- **Shared memory.** The Q tile is bf16, pre-scaled by 1/√HD. K and V are double-buffered with
  `cp.async.cg` 16-byte copies: tile i+1 is in flight while tile i computes. Every row is padded
  by 8 bf16 (`FA_LD = HD + 8`), which keeps the 32-bit fragment loads and the ldmatrix row reads
  free of bank conflicts. FA2 gets the same effect from an XOR swizzle; padding is simpler to
  address. Total: (64 + 4·32)·264·2 = 101,376 B dynamic, two blocks per SM on sm_90.
- **Online softmax.** Each accumulator row is spread over a lane quad, so the row max and sum use
  `shfl_xor` by 1 and 2. A row that has no live key yet keeps m=-inf and subtracts 0, so the
  result is 0 rather than NaN. Masked K/V slots are zero-filled rather than left stale, because a
  P of 0 times a NaN V would still poison O.
- **Causal skipping.** Positions rise with the row, so the tile's last vector has the tile's
  highest position. Keys past it are never loaded, which skips every fully masked tile. Only the
  diagonal key tile masks per element (`key > pos(v)`).
- **Key splits (flash-decoding style).** A 256-row window is only 32 query tiles × 2 kv groups =
  64 blocks, which doesn't fill 132 SMs. So `blockIdx.z` splits the window's keys
  `[0, last_pos]` into `nsplit` spans, each a multiple of 32 keys.
  - `fa2_splits` targets about 2 waves of blocks, keeps every split at 256 keys or more, and
    caps the count at what the partials buffer (`Scratch::part`) holds.
  - Each split writes `(m, l, acc[HD])` in `cx_attn_part`'s exact layout with `nchunk = nsplit`,
    so the existing `cx_attn_merge` merges the splits and applies the gate. A split past a row's
    position writes l=0, which the merge skips.
  - At 54k context the merge reads ~5 splits × 64 rows × 16 heads × 258 floats per window, which
    is negligible.
- **Numerics.** Q and P are rounded to bf16 for the MMAs; K and V are already bf16. Accumulation
  is f32. This is not split-K's summation order, so the result is not bitwise equal to narrow.
  The bar is the same one the tensor-core expert GEMM already meets: same greedy tokens and
  teacher-forced KL ≤ 1e-3.

## Expected speedup (estimate, not measured)

Causal attention FLOPs per full-attention layer are about `2·T²·HD·NH`, with 1 FMA = 2 FLOPs
and half the T×T matrix masked. Over 12 layers at T=54k that is ≈ 2.9·10¹⁴ FLOPs. With FA2-class
mma.sync on an H100/H200 at 30-50% of the ~990 TFLOP/s dense bf16 peak (FA2 reaches ~35-50% on
H100; this kernel is less tuned), attention comes to ~0.6-1.0 s at 54k.

| context | today (split-K) | attention today (≈65%) | fa2 attention (est.) | total (est.) |
|---|---|---|---|---|
| 8k  | 7.05 s | ~4.6 s | ~0.1-0.3 s | ~2.6-2.8 s (~2.6×) |
| 30k | 36.8 s | ~24 s  | ~0.3-0.8 s | ~13-14 s (~2.7×) |
| 54k | 87.2 s | ~57 s  | ~0.8-2 s   | ~31-32 s (~2.8×) |

Once attention is gone, the rest of prefill (MoE GEMMs, GDN) is the floor, so totals land near
2.6-2.8×. That matches the survey's ~2.8× at 30k.

The main risk is register pressure. 128 accumulator registers plus fragments should fit in the
255 that `__launch_bounds__(128)` allows. If ptxas spills, check with `cuobjdump -res-usage` or
NVRTC's `-Xptxas -v`. The fix would be 8 warps with Br=128, or splitting HD across two passes.

## Tests

- `cargo test -p tang-llm --lib fa2` runs on any host without CUDA:
  - `fa2_model_matches_naive_attention`: a CPU model of the kernel's arithmetic against f64
    attention. It uses the same vector packing, split spans, 32-key tiles, masking, online
    softmax, bf16 rounding of Q and P, and merge. Cases cover G=8 and G=4, partial tiles,
    windows that start mid-tile, and splits that come out empty.
  - `fa2_splits_fill_the_gpu`: the split planner.
- The fragment layouts (mma/ldmatrix lane mapping) can only be checked on a GPU, by the gate below.

## Bench and gate (H200)

Build:

```sh
cargo build --release -p tang-llm --no-default-features --features cuda
```

Run both variants at 8k / 30k / 54k. `MODEL` is the first GGUF shard of Qwen3-Coder-Next
(UD-Q4_K_XL). `TANG_CODER_WIDE` sets the wide window width (256 is the default, written out here
for clarity). `TANG_CODER_PROF=1` adds per-stage GPU time, where `attn_fa2` + `attn_merge`
replaces `attn_part` + `attn_merge`.

```sh
MODEL=/workspace/models/Qwen3-Coder-Next-UD-Q4_K_XL-00001-of-00002.gguf
# baseline: split-K
TANG_CODER_WIDE=256 TANG_WIDE_CTXS=8000,30000,54000 TANG_CODER_PROF=1 \
  ./target/release/tang-llm flash-wide-check "$MODEL" -n 64 2>&1 | tee splitk.log
# FA2
TANG_CODER_ATTN=fa2 TANG_CODER_WIDE=256 TANG_WIDE_CTXS=8000,30000,54000 TANG_CODER_PROF=1 \
  ./target/release/tang-llm flash-wide-check "$MODEL" -n 64 2>&1 | tee fa2.log
grep -E "^prefill|^wide-check|max KL" splitk.log fa2.log
```

Gates:

- Every `wide-check <len>:` line in `fa2.log` says `PASS (greedy equal, KL <= <bar>)`. That means
  the next token is equal, all 64 greedy decoded tokens are equal to narrow, and the
  teacher-forced max KL(narrow ‖ wide) is under the bar: 3.2e-5 at ≥ 8000 tokens, 2.0e-4 at
  ≥ 3000 (the fp16 FA2 bars, docs/plans/gates-2026-11-04.md), 1e-3 below that. The command exits non-zero on any FAIL.
- Compare the `wide … warm` times on the `prefill <len> tokens:` lines between the two logs. To
  make fa2 the default, it needs a clear win at 30k and 54k and no loss at 8k.

If a gate fails, `TANG_CODER_PROF` shows whether attention ran at all. An out-of-range
partials cap falls back to split-K silently. Next, rerun at `TANG_WIDE_CTXS=1000` to separate a
layout bug (wrong at any length) from a split/merge bug (wrong only once `nsplit > 1`, i.e. past
about 512 keys).

## Unverified

Nothing here has run on a GPU. This Mac has no `nvcc`, and `cudarc`'s build script fails without
CUDA, so `cargo check --features cuda` can't run here. The CUDA source (compiled at runtime by
NVRTC) and the cuda-gated Rust were only checked by reading them. `cargo check -p tang-llm`
(default features) and the CPU tests pass.
