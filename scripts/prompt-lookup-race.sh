#!/usr/bin/env bash
# Prompt-lookup speculation on Qwen3-8B through the derived k-token window.
#
# What it runs:
#   1. build qwen_decode (release, --features cuda)
#   2. tiny --derived: decode steps and a 6-token window on the GPU vs the
#      interpreter (a window of <= TANG_WINDOW_GEMV=8 rows is now lowered
#      like decode: packed-weight GEMVs with the rows in registers)
#   3. Qwen3-8B MLX 4-bit, derived: device time of a step and of windows of
#      4 and 8 at $LENS, with the short-window lowering and without it
#      (TANG_WINDOW_GEMV=0: the old f16-dequant + mma path)
#   4. per prompt (4 code, 1 prose): greedy decode of $GEN tokens, then the
#      same with prompt-lookup speculation, k = 4 and 8: tok/s end to end,
#      tokens per window, acceptance, and whether the ids equal greedy's
#
# Run from the repo root on the pod:  bash scripts/prompt-lookup-race.sh
# Env (defaults in brackets):
#   MODEL_DIR  [/root/models/Qwen3-8B-4bit]  container disk, not a network volume
#   OUT        [/root/prompt-lookup]         logs, summary, and DONE when finished
#   LENS       [128,4096]
#   CTX        [4352]
#   GEN        [128]
set -euo pipefail

REPO=${REPO:-$PWD}
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3-8B-4bit}
OUT=${OUT:-/root/prompt-lookup}
LENS=${LENS:-128,4096}
CTX=${CTX:-4352}
GEN=${GEN:-128}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
trap 'echo "exit $?" > "$OUT/DONE"' EXIT
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

log "weights: mlx-community/Qwen3-8B-4bit -> $MODEL_DIR"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q "huggingface_hub[hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  if command -v hf >/dev/null; then
    hf download mlx-community/Qwen3-8B-4bit --local-dir "$MODEL_DIR"
  else
    huggingface-cli download mlx-community/Qwen3-8B-4bit --local-dir "$MODEL_DIR"
  fi
fi

log "build"
cargo build --release -p tang-bench --features cuda --example qwen_decode 2>&1 | tail -3
BIN=target/release/examples/qwen_decode

log "tiny --derived: GPU vs interpreter"
"$BIN" tiny --derived 2>&1 | tee "$OUT/tiny-derived.txt" | tail -3 || echo "tiny derived: exit status nonzero"

P=bench/results/prompt-lookup/prompts
log "Qwen3-8B: step and window device time at $LENS"
for gemv in 8 0; do
  for w in 4 8; do
    TANG_WINDOW_GEMV=$gemv "$BIN" model "$MODEL_DIR" "$P/prose_01_sky.ids" --derived --gen 8 \
      --ctx "$CTX" --bench "$LENS" --bench-window "$w" 2>&1 | tee "$OUT/bench-gemv$gemv-w$w.txt" \
      | grep -E '^(decode step at len|window of)' || true
  done
done

log "Qwen3-8B: greedy vs prompt lookup, $GEN tokens"
for ids in "$P"/*.ids; do
  name=$(basename "$ids" .ids)
  for k in 4 8; do
    "$BIN" model "$MODEL_DIR" "$ids" --derived --gen "$GEN" --ctx "$CTX" --spec "$k" --ngram 4 \
      2>&1 | tee "$OUT/spec-$name-k$k.txt" | grep -E '^(decode .* end to end|spec k=)' || true
  done
done

log "summary"
{
  echo "GPU: $(tail -1 "$OUT/gpu.txt")"
  echo "commit: $(cat "$OUT/commit.txt")"
  echo
  echo "tiny --derived: $(grep -E '^tiny' "$OUT/tiny-derived.txt" | tr '\n' ' ')"
  echo
  echo "device time (TANG_WINDOW_GEMV=8 new, =0 old):"
  for f in "$OUT"/bench-*.txt; do
    grep -E '^(decode step at len|window of)' "$f" | sed "s|^|  [$(basename "$f" .txt)] |"
  done
  echo
  echo "prompt lookup:"
  for f in "$OUT"/spec-*.txt; do
    echo "  $(basename "$f" .txt):"
    grep -E '^decode .* end to end' "$f" | sed 's/^/    greedy: /'
    grep -E '^spec k=[0-9]+: ([0-9]+ tokens|first)' "$f" | sed 's/^/    /'
  done
} | tee "$OUT/summary.txt"
