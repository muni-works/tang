#!/usr/bin/env bash
# Every hand-vs-derived race on this pod, one board row per (model, context).
#
#   qwen3-8b    scripts/derived-race.sh         /models/Qwen3-8B-4bit
#   coder-next  scripts/coder-derived-race.sh   /models/coder-next
#   ds          scripts/ds-derived-race.sh      /models/deepseek-v4.1   (skipped until both exist)
#   glm         scripts/glm-derived-race.sh     /models/glm-5.3         (skipped until both exist)
#
# Weights come from /models (the RunPod global volume); a model dir is used
# only when it has a .complete marker. A missing model or race script is a
# skip, not a failure. Each race's rows (scripts/bench_board.py parse) are
# appended to bench/results/board.jsonl; open bench/board/index.html to see them.
#
# Run from the repo root:  bash scripts/bench-all.sh
# Env: MODELS [/models], OUT [/root/bench-all], BOARD [bench/results/board.jsonl],
#      ONLY (comma list of names, e.g. ONLY=qwen3-8b), LENS, CTX (passed through).
# Writes $OUT/DONE when finished (wait loops: until [ -f $OUT/DONE ]); its
# text lists each model as ok, skip or fail.
set -uo pipefail

REPO=${REPO:-$PWD}
MODELS=${MODELS:-/models}
OUT=${OUT:-/root/bench-all}
BOARD=${BOARD:-$REPO/bench/results/board.jsonl}
ONLY=${ONLY:-}
cd "$REPO"
mkdir -p "$OUT" "$(dirname "$BOARD")"
rm -f "$OUT/DONE"
touch "$BOARD"
: >"$OUT/status.txt"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)

status() { echo "$1: $2" | tee -a "$OUT/status.txt"; }

# name  race-script  model-dir  file-or-dir-the-race-reads (relative to model dir)
race() {
  local name=$1 script=$2 dir=$3
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then return; fi
  if [ ! -f "$script" ]; then status "$name" "skip (no $script)"; return; fi
  if [ ! -f "$dir/.complete" ]; then status "$name" "skip (no $dir/.complete)"; return; fi
  local out="$OUT/$name"
  rm -rf "$out"
  mkdir -p "$out"
  printf '\n=== %s: %s\n' "$name" "$script"
  if ! MODEL_DIR="$dir" OUT="$out" bash "$script" 2>&1 | tee "$out/race.log"; then
    status "$name" "fail (race exited nonzero, see $out/race.log)"
  fi
  local rows
  if rows=$(python3 scripts/bench_board.py parse "$out" --model "$name" --branch "$BRANCH"); then
    printf '%s\n' "$rows" >>"$BOARD"
    grep -q "^$name: fail" "$OUT/status.txt" || status "$name" "ok ($(printf '%s\n' "$rows" | wc -l | tr -d ' ') rows)"
  else
    status "$name" "fail (no rows parsed from $out)"
  fi
}

race qwen3-8b   scripts/derived-race.sh       "$MODELS/Qwen3-8B-4bit"
# coder-derived-race.sh: SKIP_REF=1 by default here (flash-ref is a 20-40 min CPU check, not a speed number)
export SKIP_REF=${SKIP_REF:-1}
race coder-next scripts/coder-derived-race.sh "$MODELS/coder-next"
race ds         scripts/ds-derived-race.sh    "$MODELS/deepseek-v4.1"
race glm        scripts/glm-derived-race.sh   "$MODELS/glm-5.3"

printf '\n=== board: %s (%s rows)\n' "$BOARD" "$(wc -l <"$BOARD" | tr -d ' ')"
cat "$OUT/status.txt"
{ echo "DONE bench-all $(date -u +%FT%TZ)"; cat "$OUT/status.txt"; } >"$OUT/DONE"
