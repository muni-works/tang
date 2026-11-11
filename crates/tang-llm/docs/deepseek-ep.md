# DeepSeek-V4.1-Flash: expert parallelism

M3 of [deepseek-v41-plan.md](deepseek-v41-plan.md). The routed experts are 40 × 384 × 3
fp4 matrices, ≈ 272 GB: no single GPU holds them, so one GPU streams them from host through
the expert LRU (`DS_EXPERT_SLOTS`) at ~1 tok/s. EP puts every expert in VRAM by sharding them
across N GPUs in one process. Code: `src/flash/deepseek/ep.rs`, `tang-moe/src/nccl.rs`.

## Layout

- **Experts sharded by id**: expert `e` lives on rank `e % N` (`ep::owner`). A rank's
  `GpuModel::owns_expert` filter skips the others in `moe`, so their weights are never
  uploaded. 8 × H200 holds 272 / 8 = 34 GB of experts each; 4 × B200, 68 GB.
- **Everything else replicated**: embeddings, attention, compressor/indexer, mHC, router,
  shared expert, head, DSpark, and Engram's gate (the tables stay mapped on the host). These
  are small next to the experts and keep every rank's forward identical outside the MoE.
- **One thread per GPU**, each with its own primary context bound (`GpuModel::open_on`) and
  its own blocking stream. `GpuModel` is not `Send`, so each is built and used on its thread.

## A step

`EpGroup` sends each command (forward, DSpark draft, reset) to every rank over a channel and
waits for all replies: that is the step barrier. The token ids live in host memory in one
process, so the "broadcast from rank 0" is the channel; `Comm::broadcast` exists for when
ranks are separate processes.

Each rank runs the whole forward on the same tokens. In `moe`:

1. Router on every rank (replicated; same inputs, same kernel, same top-k).
2. Each rank runs its local experts on the tokens routed to them and scatter-adds into a zeroed
   `[t][d]` f32 buffer `y`, exactly as the reference accumulates `y`.
3. `moe_reduce`: all-reduce (sum) of `y` across ranks on each rank's stream (NCCL), so every
   rank holds the same full routed sum.
4. Every rank adds the shared expert to it, identically (`ds_add_round`), once.

All-reduce rather than all-to-all dispatch/combine: at batch 1–8 `y` is ≤ 8 × 4096 × 4 B =
128 KB a layer, so the all-reduce is latency-bound (~10–20 µs on NVLink, ×40 layers) and
needs no token permutation. All-to-all wins at prefill / large batch; it slots into the same
hook later. Rank 0 owns sampling and reads the logits; the other ranks compute the same
logits and discard them (the head is replicated; cheaper than syncing hidden states).

## Backends

- `Backend::Nccl` (cargo feature `ep` = `tang-moe/nccl`, off by default): `ncclAllReduce`
  in place. cudarc dlopens `libnccl.so.2`, so the build needs no NCCL; `EpGroup::open` errors
  without it. NCCL needs distinct devices per rank.
- `Backend::Host`: read back, sum in rank order, write back, behind a `Barrier`. Slow but
  deterministic, allows several ranks on one device, and is what the tests run on mew.

## Determinism

The unsharded engine adds expert outputs into `y` in expert order. Sharded, each rank sums
its own experts, then the reduce adds the partials: a different f32 summation order, so the
logits differ in the last bits. NCCL's ring/tree order can also change with world size and
topology (all ranks still get the same bits from one all-reduce, so they stay in lockstep).
The parity bar is the plan's: KL and top-1 against the reference, not bit-exactness. World 1
is bit-exact with the single-GPU engine (`ep_world1_is_exact`).

## Caveats

- If a rank errors mid-forward, the others block in the collective; a step is all-or-nothing.
- Not yet: CUDA graphs per rank, overlapping the all-reduce with the shared expert (it is
  independent of `y`), and the host-side routing lists moving to the device.

## Tests

- `ep_world1_is_exact`: one rank, host reduce, equals `GpuModel` bit for bit.
- `ep_two_ranks_one_device_match`: two ranks on one GPU with even/odd experts, host reduce:
  KL < 1e-4 and same top-1 against the unsharded engine; every expert uploaded exactly once.
- `tang_moe::nccl::tests::nccl_all_reduce_and_broadcast` and `ep_nccl_matches_single_gpu`
  (`#[ignore]`): need ≥ 2 GPUs and libnccl. On a multi-GPU pod:

```
cargo test --release -p tang-moe --features nccl --lib nccl -- --ignored
cd crates/tang-llm && DS41_TINY=/tmp/ds41-tiny EP_WORLD=2 TANG_REQUIRE_CUDA=1 \
  cargo test --release --no-default-features --features ep --lib flash::deepseek::ep -- --ignored --test-threads 1
```
