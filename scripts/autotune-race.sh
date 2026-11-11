#!/usr/bin/env bash
# The schedule autotuner on one model (warm runner: race:<model>:scripts/autotune-race.sh):
#   1. baseline: the derived step with no tuning cache (TANG_TUNE=0): greedy ids + step ms
#   2. --tune: sweep, select per kernel signature, write $OUT/tune.tsv
#   3. tuned: the same run reading $OUT/tune.tsv: greedy ids + step ms
#   4. qwen3-8b: the hand step too (step ms, ids)
# Env: MODEL_DIR, OUT, LENS [128,4096], CTX [4352], TUNE_REPS [10]
set -euo pipefail
OUT=${OUT:-/root/autotune}; LENS=${LENS:-128,4096}; CTX=${CTX:-4352}
mkdir -p "$OUT"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$OUT/gpu.txt"
git log --oneline -1 | tee "$OUT/commit.txt"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
  model=qwen3-8b
  cargo build --release -p tang-bench --features cuda --example qwen_decode 2>&1 | tail -1
  echo "785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 13 576 6722 315 15344 374 21718 13" >"$OUT/prompt.ids"
  run() { target/release/examples/qwen_decode model "$MODEL_DIR" "$OUT/prompt.ids" --gen 64 --ctx "$CTX" "$@"; }
  echo "=== hand"; TANG_TUNE=0 run --bench "$LENS" --out "$OUT/ids-hand.txt" 2>&1 | grep -E 'decode step|tok/s|ids' | tee "$OUT/hand.txt"
  D=--derived
else
  model=coder-next
  cargo build --release -p tang-bench --features cuda,gguf --example coder_derived 2>&1 | tail -1
  echo "785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 13 576 6722 315 15344 374 21718 13" >"$OUT/prompt.ids"
  run() { target/release/examples/coder_derived "$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf" --ids-file "$OUT/prompt.ids" -n 64 --ctx "$CTX" "$@"; }
  D=
fi
echo "=== baseline (no cache)"
TANG_TUNE=0 run $D --bench "$LENS" --out "$OUT/ids-base.txt" 2>&1 | grep -E 'step|tok/s|cache' | tee "$OUT/base.txt"
echo "=== tune"
TANG_TUNE=0 TANG_TUNE_OUT="$OUT/tune.tsv" run $D --tune 2>&1 | grep -E 'autotune|tuning|^  ' | tee "$OUT/tune.txt"
echo "=== tuned"
TANG_TUNE_CACHE="$OUT/tune.tsv" run $D --bench "$LENS" --out "$OUT/ids-tuned.txt" 2>&1 | grep -E 'step|tok/s|cache' | tee "$OUT/tuned.txt"
cmp -s "$OUT/ids-base.txt" "$OUT/ids-tuned.txt" && echo "greedy ids tuned vs base: identical" || echo "greedy ids tuned vs base: DIFFER"
[ -f "$OUT/ids-hand.txt" ] && { cmp -s "$OUT/ids-hand.txt" "$OUT/ids-tuned.txt" && echo "greedy ids tuned vs hand: identical" || echo "greedy ids tuned vs hand: differ"; }
cat "$OUT/tune.tsv"
