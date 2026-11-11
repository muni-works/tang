#!/usr/bin/env bash
# Prefix cache under an agent-like trace: tang-llm serve (coder-next, --batch) replayed by
# scripts/prefix-replay.py once per mode, a fresh server and state dir each:
#   off    TANG_SERVE_CACHE=off: nothing reused (TTFT without the cache)
#   old    TANG_SERVE_SPILL=0, --snapshots 6, 12 GB: the policy before this branch
#   new    spill evicted slots to host RAM, rewind parked slots to saved GDN states
# Run on a warm runner:  scripts/gpu-run.sh <pod> <ref> race:coder-next:scripts/prefix-replay.sh
# Env: MODEL_DIR, OUT, MODES [off old new], BATCH [16], CTX [32768], plus the replay's
# SESSIONS/TURNS/CONC/SYS_TOK/FIRST_TOK/TOOL_TOK/MAX_TOKENS/REWIND_EVERY.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/root/runner/models/coder-next}
OUT=${OUT:-/root/prefix-replay}
MODES=${MODES:-off old new}
BATCH=${BATCH:-16}
CTX=${CTX:-32768}
GGUF="$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
mkdir -p "$OUT"
export TANG_CODER_KV=${TANG_CODER_KV:-int8}

cargo build --release -p tang-llm --features cuda --bin tang-llm 2>&1 | tail -2
LLM=${CARGO_TARGET_DIR:-target}/release/tang-llm
free -g | head -2 | tee "$OUT/mem.txt"

for mode in $MODES; do
  state=$(mktemp -d)
  extra=()
  envs=(TANG_FLASH_STATE_DIR="$state" TANG_SNAP_TIMES=1)
  case $mode in
    off) envs+=(TANG_SERVE_CACHE=off) ;;
    old) envs+=(TANG_SERVE_SPILL=0 TANG_FLASH_SNAP_GB=12); extra=(--snapshots 6) ;;
    new) ;;
  esac
  log="$OUT/serve-$mode.log"
  env "${envs[@]}" "$LLM" serve "$GGUF" --port 8931 --ctx "$CTX" --batch "$BATCH" \
    "${extra[@]}" >"$log" 2>&1 &
  pid=$!
  for _ in $(seq 600); do
    curl -sf localhost:8931/v1/models >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { tail -30 "$log"; exit 1; }
    sleep 2
  done
  # VRAM once loaded, before the run grows the host cache.
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/vram-$mode.txt"
  LABEL=$mode OUT=$OUT URL=http://127.0.0.1:8931 python3 scripts/prefix-replay.py | tee -a "$OUT/summary.txt"
  grep -E "sequence slots fit|cache:" "$log" | tail -2 | tee -a "$OUT/summary.txt"
  grep -c "spilled slot" "$log" | sed 's/^/spills logged: /' | tee -a "$OUT/summary.txt"
  ps -o rss= -p $pid | awk '{printf "server RSS %.1f GB\n", $1/1048576}' | tee -a "$OUT/summary.txt"
  kill $pid; wait $pid 2>/dev/null || true
  rm -rf "$state"
done
