#!/usr/bin/env bash
# Tiny-model check: run it before any full race. A few minutes when the build is warm.
#
#   1. qwen_decode tiny, hand and derived: GPU vs the interpreter (parity)
#   2. Qwen3-8B 4-bit: 21-token prompt, $GEN greedy tokens, device step time at $LENS,
#      hand and derived; greedy ids compared
#
# Writes $OUT/{gpu,commit,tiny-*,model-*,summary}.txt in the layout
# scripts/bench_board.py parse reads. Exit status nonzero if tiny or ids fail.
# Env: OUT, MODEL_DIR [/models/qwen3-8b-4bit], LENS [128], GEN [16], CTX [1024], BIN
set -uo pipefail
OUT=${OUT:?}
MODEL_DIR=${MODEL_DIR:-/models/qwen3-8b-4bit}
LENS=${LENS:-128}
GEN=${GEN:-16}
CTX=${CTX:-1024}
BIN=${BIN:-target/release/examples/qwen_decode}
mkdir -p "$OUT"
rc=0

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >"$OUT/gpu.txt"
{ git log --oneline -1 2>/dev/null || echo "no git checkout"; } >"$OUT/commit.txt"

for kind in hand derived; do
  flag=; [ $kind = derived ] && flag=--derived
  # shellcheck disable=SC2086
  if ! "$BIN" tiny $flag >"$OUT/tiny-$kind.txt" 2>&1; then
    echo "tiny $kind: FAIL"; rc=1
  fi
  tail -2 "$OUT/tiny-$kind.txt"
done

echo "785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 13 576 6722 315 15344 374 21718 13" \
  >"$OUT/prompt.ids"
for kind in hand derived; do
  flag=; [ $kind = derived ] && flag=--derived
  # shellcheck disable=SC2086
  "$BIN" model "$MODEL_DIR" "$OUT/prompt.ids" --gen "$GEN" --ctx "$CTX" --bench "$LENS" \
    --out "$OUT/ids-$kind.txt" $flag >"$OUT/model-$kind.txt" 2>&1 || { echo "model $kind: FAIL"; rc=1; }
  grep -E '^decode step at len' "$OUT/model-$kind.txt" | sed "s/^/  [$kind] /"
done

if cmp -s "$OUT/ids-hand.txt" "$OUT/ids-derived.txt"; then
  ids="greedy ids, derived vs hand: identical ($GEN tokens)"
else
  ids="greedy ids, derived vs hand: differ"; rc=1
fi
{
  echo "GPU: $(cut -d, -f1 "$OUT/gpu.txt")"
  echo "commit: $(cut -d' ' -f1 "$OUT/commit.txt")"
  for kind in hand derived; do echo "tiny $kind: $(grep -E '^tiny' "$OUT/tiny-$kind.txt" | tr '\n' ' ')"; done
  for kind in hand derived; do grep -E '^decode step at len' "$OUT/model-$kind.txt"; done
  echo "$ids"
} >"$OUT/summary.txt"
cat "$OUT/summary.txt"
exit $rc
