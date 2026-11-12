#!/usr/bin/env bash
# Qwen3-Coder-Next batched decode from the derived step (examples/coder_batched.rs):
# B streams at ragged lengths stepped together, routed experts grouped by expert
# across streams, int8 contractions staged by positions.
#
#   1. parity: S streams teacher-forced from position 0, the last logits of the
#      batched run (B = S) against B = 1 running the S streams one at a time
#   2. timing: device ms per step and aggregate tok/s at each B in $BATCHES,
#      streams ragged around each length in $LENS
#
# Env: MODEL_DIR [/root/models/coder-next], OUT [/root/coder-batched],
# BATCHES [1,4,16,64], LENS [128,1024], CTX [2048], CHECK [4,16] (S values),
# CHECK_STEPS [24], plus TANG_BATCH_* knobs (see BlockOpts::batched).
set -euo pipefail
REPO=${REPO:-$PWD}
MODEL_DIR=${MODEL_DIR:-/root/models/coder-next}
OUT=${OUT:-/root/coder-batched}
BATCHES=${BATCHES:-1,4,16,64}
LENS=${LENS:-128,1024}
CTX=${CTX:-2048}
CHECK=${CHECK:-4,16}
CHECK_STEPS=${CHECK_STEPS:-24}
GGUF="$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
mkdir -p "$OUT"
cd "$REPO"
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
git log --oneline -1 2>/dev/null | tee "$OUT/commit.txt" || true
cargo build --release -p tang-bench --features cuda,gguf --example coder_batched 2>&1 | tail -2
BIN=${CARGO_TARGET_DIR:-target}/release/examples/coder_batched
# a fixed token sequence (any ids: teacher-forced)
python3 -c 'print(" ".join(str((7919 * i + 151) % 150000) for i in range(64)))' > "$OUT/ids.txt"

for S in ${CHECK//,/ }; do
  echo "=== parity S=$S"
  "$BIN" "$GGUF" --batch 1 --streams "$S" --ctx "$CTX" --check-ids "$OUT/ids.txt" \
    --check-steps "$CHECK_STEPS" --check-out "$OUT/ref$S.txt" 2>&1 | grep -v "^$" | tail -3
  "$BIN" "$GGUF" --batch "$S" --ctx "$CTX" --check-ids "$OUT/ids.txt" \
    --check-steps "$CHECK_STEPS" --check-in "$OUT/ref$S.txt" 2>&1 | tail -4
done
for B in ${BATCHES//,/ }; do
  echo "=== timing B=$B"
  "$BIN" "$GGUF" --batch "$B" --ctx "$CTX" --lens "$LENS" 2>&1 | grep -E "batched|expert|weights|panick|error" | tee -a "$OUT/timing.txt"
done
