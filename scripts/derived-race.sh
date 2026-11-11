#!/usr/bin/env bash
# Derived vs hand-written Qwen3 steps on one CUDA GPU (PR #205).
#
# The hand step is ir_qwen::step_graph (#56). The derived step comes from the
# one prefill graph (ir_qwen::prefill) through tang_compile::tensor::step:
# `qwen_decode ... --derived`. Both use the same passes (compiled_decode) and
# the same lowering (to_cuda_decode).
#
# What it runs:
#   1. build qwen_decode (release, --features cuda)
#   2. tiny random model on the GPU vs the interpreter: hand, derived; f16
#      and f32 caches
#   3. Qwen3-8B MLX 4-bit: a 21-token prompt, 64 greedy tokens, then device
#      time of a decode step (k = 1) and a 4-token window (k = 4) at each
#      length in $LENS. Hand first, then derived.
#   4. a summary: step ms and tok/s side by side, and whether the greedy
#      ids agree
#
# Run from the repo root on the pod:  bash scripts/derived-race.sh
# Env (defaults in brackets):
#   MODEL_DIR  [/root/models/Qwen3-8B-4bit]  container disk, not a network volume
#   OUT        [/root/derived-race]          logs and the summary
#   LENS       [128,4096]                    context lengths for the bench
#   CTX        [4352]                        KV capacity
#   WINDOW     [4]                           window size for the k > 1 bench
#
# The bench numbers are device time only (Program::time, best of 3x20 replays),
# so the derived step's host-side token write and argmax don't count. They do
# count in the "end to end" decode line, which is not a like-for-like
# comparison: the hand step keeps the next token and position on the device.
set -euo pipefail

REPO=${REPO:-$PWD}
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3-8B-4bit}
OUT=${OUT:-/root/derived-race}
LENS=${LENS:-128,4096}
CTX=${CTX:-4352}
WINDOW=${WINDOW:-4}
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
# (a copy without .git, e.g. rsynced, has no commit to name)
{ git log --oneline -1 2>/dev/null || echo "no git checkout"; } | tee "$OUT/commit.txt"

log "weights: mlx-community/Qwen3-8B-4bit -> $MODEL_DIR"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  # the RunPod PyTorch image's system Python is "externally managed"
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q "huggingface_hub[hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  if command -v hf >/dev/null; then
    hf download mlx-community/Qwen3-8B-4bit --local-dir "$MODEL_DIR"
  else
    huggingface-cli download mlx-community/Qwen3-8B-4bit --local-dir "$MODEL_DIR"
  fi
fi
test -f "$MODEL_DIR/model.safetensors" || {
  echo "no $MODEL_DIR/model.safetensors (qwen_decode reads one file)" >&2
  exit 1
}

log "build"
time cargo build --release -p tang-bench --features cuda --example qwen_decode 2>&1 | tail -3
BIN=target/release/examples/qwen_decode

log "tiny: GPU vs interpreter"
for kv in f16 f32; do
  if [ "$kv" = f32 ]; then export TANG_KV_F32=1; else unset TANG_KV_F32; fi
  # a failed check is reported and the race goes on
  "$BIN" tiny 2>&1 | tee "$OUT/tiny-hand-$kv.txt" | tail -2 || echo "tiny hand $kv: exit status nonzero"
  "$BIN" tiny --derived 2>&1 | tee "$OUT/tiny-derived-$kv.txt" | tail -2 || echo "tiny derived $kv: exit status nonzero"
done
unset TANG_KV_F32

log "Qwen3-8B: prompt, greedy decode, bench (k = 1 and k = $WINDOW) at $LENS"
# a fixed 21-token prompt (any ids will do for timing; both runs use the same)
echo "785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 13 576 6722 315 15344 374 21718 13" \
  >"$OUT/prompt.ids"
for kind in hand derived; do
  flag=""
  [ "$kind" = derived ] && flag="--derived"
  # shellcheck disable=SC2086
  "$BIN" model "$MODEL_DIR" "$OUT/prompt.ids" --gen 64 --ctx "$CTX" \
    --bench "$LENS" --bench-window "$WINDOW" --out "$OUT/ids-$kind.txt" $flag \
    2>&1 | tee "$OUT/model-$kind.txt"
done

log "summary"
{
  echo "GPU: $(tail -1 "$OUT/gpu.txt")"
  echo "commit: $(cat "$OUT/commit.txt")"
  echo
  echo "tiny (worst relative logit error, GPU vs interpreter):"
  for kv in f16 f32; do
    for kind in hand derived; do
      printf '  %-8s %s: %s\n' "$kind" "$kv" "$(grep -E '^tiny' "$OUT/tiny-$kind-$kv.txt" | tr '\n' ' ')"
    done
  done
  echo
  echo "Qwen3-8B device step time:"
  for kind in hand derived; do
    grep -E '^(decode step at len|window of)' "$OUT/model-$kind.txt" | sed 's/^/  /'
  done
  echo
  echo "end to end (not like for like, see the header):"
  for kind in hand derived; do
    printf '  %-8s %s\n' "$kind" "$(grep -E '^decode .* tok/s end to end' "$OUT/model-$kind.txt")"
  done
  echo
  if cmp -s "$OUT/ids-hand.txt" "$OUT/ids-derived.txt"; then
    echo "greedy ids: identical (64 tokens)"
  else
    echo "greedy ids differ:"
    echo "  hand:    $(cat "$OUT/ids-hand.txt")"
    echo "  derived: $(cat "$OUT/ids-derived.txt")"
  fi
} | tee "$OUT/summary.txt"
