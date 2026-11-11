# glm5 wiring — the resident engine (frog/glm5-wiring)

The engine wired in: **DevKda per layer** (W4's core), **MhcGpu per layer** (W2), and
**the KDA projections as GPU GEMVs** (W5's G5W pattern, Q8_0-native). This was the last
engine gate between the tree and production: every GPU struct was built and tested but
the engine still called the free CPU functions (22 s/window on the real model).

## What's wired (all committed on frog/glm5-wiring)

| Step | Commit | What |
|---|---|---|
| DevLayer (kda_dev.rs) | `c7c73ae9` | one struct per KDA layer: DevKda (conv+state resident) + q/k/v/o/f/g/b_proj as `k5_gemv` on G5W views of the raw GGUF bytes (Q8_0 stays quantised, 67 MB/layer), gate_finish/beta/gated_norm kernels (kda_k5.cuh) |
| engine wiring | `220d6d57` | GpuFfn::load builds a DevLayer per KDA layer; kda_mixer routes through it (device owns the state; host sync only across fallback paths); `wc.0.clone()` (~268 MB/window) killed |
| MhcGpu wiring | `911244b3` | one MhcGpu per layer at load (both blocks + scratch pool); mhc_pre/mhc_post call `MhcGpu::pre`/`post_mixes`; the 7-alloc staging path deleted |
| fallbacks + cuda check | `58d73e1e` | windows > max_t fall to CPU mHC / staged KDA; `--features cuda` type-checks clean |
| wiring regression tests | `0e206a1c` | **THE LESSON pinned as tests**: resident layers must be CONSTRUCTED at `use_gpu()` and CALLED (device-fresh flag flips) |
| host kernel check | `b3f8bd06` | kda_k5.cuh compiles+runs as host C++, 8/8 checks vs independent references |

Key files: `crates/tang-llm/src/flash/glm5/kda_dev.rs` (DevLayer, ~440 LOC),
`kda_k5.cuh` (the 4 kernels), `moe_gpu.rs::Kernels5`, engine.rs (GpuFfn + kda_mixer +
mhc_pre/post).

Design decisions worth knowing:

- **Weights stay GGUF-format on the device** (Q8_0): `kda_dev::dev_w` wraps
  `moe::gpu::upload`, the kernel dequantises element-wise (`g5_dot`). Same bytes the MoE
  path uses; 67 MB/layer resident vs 268 MB if f32.
- **The device owns the live KDA state.** The engine's host `(ConvState, KdaState)`
  is synced (`read_back`/`seed`) only across CPU-fallback windows; `kda_dev_fresh[l]`
  tracks the owner. Decode keeps the host conv copy fresh (W4's launchers do).
- **Scratch is `TANG_GLM5_KDA_MAXT` (default 512)** — the engine's prefill chunk — so a
  raw `window()` beyond it falls back rather than mis-sizing. Prefill chunks of 64 run
  W4's 64-token kernel; a <64 tail runs the CPU core on read-back state and seeds forward.
- **mHC's `MhcGpu::pre` reuses the scratch pool** and `post_mixes` repacks the engine's
  `Vec<Mix>` flat (no per-token allocs).

## mew commands (the 3090 pod)

```bash
# build (cuda, no default features — Metal is macOS-only)
cd ~/tang-scratch/glm5            # or the fresh clone/worktree to test
export PATH=/usr/local/cuda-12.8/bin:$HOME/.cargo/bin:$PATH
export CUDA_PATH=/usr/local/cuda-12.8
export CARGO_TARGET_DIR=$HOME/tang-scratch/target
cargo build --release -p tang-llm --no-default-features --features cuda

# 1. cuda-gated parity tests on the tiny model (must all pass):
TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --no-default-features \
  --features cuda --lib glm5::engine -- --nocapture
#   engine_gpu_matches_cpu        (f32 GPU paths vs CPU, 1e-3 bar)
#   engine_gpu_mla_matches_cpu    (all-GPU incl. bf16 MLA, top-1 >= 98%)
#   gpu_resident_layers_are_constructed   (NEW: the wiring regression)
#   gpu_kda_window_routes_through_resident_layer  (NEW)

# 2. the verification profile — hybrid run on the REAL model
#    (KDA/MLA/mHC on the GPU, experts on CPU/mmap; 24 GB card)
BIN=$HOME/tang-scratch/target/release/tang-llm
G=$(find ~/tang-scratch/glm53/UD-Q4_K_XL -name '*-00001-of-*.gguf' | head -1)
TANG_GLM5_PROF=1 TANG_GPUS=2 TANG_GLM5_GPU_FFN=0 timeout 1800 $BIN flash-generate "$G" \
  --prompt "Write a haiku about the sea." -n 8 --temp 0 -v 2>&1 | tee ~/tang-scratch/wiring-profile.log

# what to look for in the prof lines (per window):
#   prof kda weights   -> ~0 (was ~268 MB dequant per KDA layer; now the Arc ref only)
#   prof kda mixer     -> GPU kernel time, tens of ms total (was ~8 GFLOP scalar/layer)
#   prof mhc_pre/post  -> resident path (was 7 allocs + 1.6 MB uploads per call)
#   window t=1 ...     -> the headline: was 22 s/window pre-wiring
# generation quality: the 8 tokens must be the same greedy ids the CPU binary gives
# (run the same command without TANG_GLM5_PROF once to diff `ids:` lines — they must match
# exactly; any divergence is a wiring bug, not numerics)
```

## What needs mew (cannot be checked on this Mac)

1. **NVRTC compile of `kda_k5.cuh` against `sm_86/89/120`** — the host C++ check verifies
   logic, not NVRTC. If `k5_gemv` fails to compile, likely `G5W` struct passing by value
   in the args list (W5's kernels do the same, so it should be fine).
2. **DevKda + GEMV integration on real silicon** — the resident-path engine tests above.
3. **The real-model profile** — the verification run; compare per-stage ms against the
   pre-wiring profile in `~/tang-scratch/profile-result.md`.
4. **Multi-GPU split** (`TANG_GPUS=2 TANG_SPLIT_AT=...`): each DevLayer binds its own
   device at load and per window; the CUDA context juggling through `Gpu::bind` is the
   same pattern the MoE path uses, but the KDA state is now per-device-resident —
   worth one tiny-model run with the split to confirm no cross-device state mixups.

## CPU invariants (already verified here)

- `cargo test -p tang-llm --lib glm5` → 36/36 (incl. engine-vs-reference parity on the
  tiny model).
- `TANG_GLM5_GPU=0` is bit-identical: the CPU paths are untouched (the wiring lives
  entirely behind `#[cfg(feature = "cuda")]` guards checked at load).
- `CUDARC_CUDA_VERSION=12080 cargo check -p tang-llm --lib --no-default-features
  --features cuda` → clean.
- `crates/tang-llm/scripts/glm5_kda_k5_host.sh` → 8/8.
