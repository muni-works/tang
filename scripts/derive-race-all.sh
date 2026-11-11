#!/usr/bin/env bash
# One pod, four models: derived steps on the GPU (claude/derive-integrate).
#
#   1. tiny, GPU vs interpreter, every model: `derive_tiny_gpu 1 4` runs the
#      derived step (k = 1 and a 4-token window) of the Qwen3, coder-next,
#      DeepSeek V4.1 and GLM-5.3 tiny prefill graphs on the device against
#      the f64 interpreter, caches seeded from a CPU prefill (NaN in
#      unwritten rows), and prints device ms per step.
#   2. real size, hand vs derived: Qwen3-8B only (scripts/derived-race.sh:
#      the hand #56 step vs the derived step, decode and a 4-token window at
#      $LENS, all-rows and last-row windows, 64 greedy tokens). The other
#      three have no real-size derived path yet: their prefill graphs take
#      f32 weights (80B-700B parameters) and no quantized lowering exists
#      for them, so they run tiny only.
#
# Run from the repo root on the pod:  bash scripts/derive-race-all.sh
# Needs: one CUDA GPU (SM80+; RTX 3090/4090, A100, H100, RTX PRO 6000),
# ~40 GB container disk (Qwen3-8B MLX 4-bit ~4.6 GB, cargo target ~10 GB),
# about an hour (build 10-15 min, weights a few minutes, tiny ~2 min, the
# three Qwen3-8B runs a few minutes each).
#
# Env: OUT [/root/derive-race-all], plus derived-race.sh's MODEL_DIR, LENS,
# CTX, WINDOW. SKIP_REAL=1 runs the tiny checks only.
set -euo pipefail

REPO=${REPO:-$PWD}
OUT=${OUT:-/root/derive-race-all}
mkdir -p "$OUT"
cd "$REPO"

log() { printf '\n=== %s\n' "$*"; }

log "toolchain and GPU"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"
{ git log --oneline -1 2>/dev/null || echo "no git checkout"; } | tee "$OUT/commit.txt"

log "build"
time cargo build --release -p tang-bench --features cuda --example derive_tiny_gpu 2>&1 | tail -3

log "tiny: derived steps of all four models, GPU vs interpreter"
target/release/examples/derive_tiny_gpu 1 4 2>&1 | tee "$OUT/tiny-all.txt" ||
  echo "derive_tiny_gpu: a check failed (see $OUT/tiny-all.txt)"

if [ "${SKIP_REAL:-0}" != 1 ]; then
  log "real size: Qwen3-8B hand vs derived"
  OUT="$OUT/qwen3-8b" bash scripts/derived-race.sh ||
    echo "derived-race.sh failed (see $OUT/qwen3-8b)"
fi

log "summary"
{
  echo "GPU: $(tail -1 "$OUT/gpu.txt")"
  echo "commit: $(cat "$OUT/commit.txt")"
  echo
  echo "tiny, derived step on the GPU vs the interpreter (k = 1, 4):"
  grep -E ' k=[0-9]+: ' "$OUT/tiny-all.txt" | sed 's/^/  /'
  if [ -f "$OUT/qwen3-8b/summary.txt" ]; then
    echo
    echo "Qwen3-8B, hand vs derived:"
    sed 's/^/  /' "$OUT/qwen3-8b/summary.txt"
  fi
} | tee "$OUT/summary.txt"
