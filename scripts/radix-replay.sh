#!/usr/bin/env bash
# Cross-user prefix cache under an agent trace with shared prefixes (scripts/prefix-replay.py
# with TENANTS / FILES / FORK_EVERY): tang-llm serve (coder-next, --batch) once per mode, a
# fresh server each.
#   off        TANG_SERVE_CACHE=off: nothing reused
#   s249       #249's policy: states at template marks only, pure LRU, only pinned states on
#              disk, shared across everyone
#   radix      split points, reuse-weighted eviction, every state written to the
#              content-addressed store, shared across everyone (TANG_SERVE_SHARE=all)
#   radix-sys  radix with TANG_SERVE_SHARE=system: only the system message crosses tenants
#   <mode>+    the same mode again on a FRESH server over the previous <mode> run's state dir:
#              returning users after a pod restart (needs <mode> earlier in MODES)
# Then O_DIRECT read bandwidth of the hot state dir's disk and of COLD (if set).
# Run on a warm runner:  scripts/gpu-run.sh <pod> <ref> race:coder-next:scripts/radix-replay.sh
# Env: MODEL_DIR, OUT (the runner sets it), MODES, BATCH [16], CTX [32768], COLD (a dir on the
# network volume, for a radix-cold mode: hot budget 0, every state read from COLD), plus the
# replay's knobs (defaults below).
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/root/runner/models/coder-next}
OUT=${OUT:-/root/radix-replay}
MODES=${MODES:-s249 radix radix-sys s249+ radix+}
BATCH=${BATCH:-16}
CTX=${CTX:-32768}
GGUF="$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
mkdir -p "$OUT"
export TANG_CODER_KV=${TANG_CODER_KV:-int8}
export SESSIONS=${SESSIONS:-32} TURNS=${TURNS:-5} CONC=${CONC:-16} TENANTS=${TENANTS:-8}
export FILES=${FILES:-6} FILE_TOK=${FILE_TOK:-2000} FORK_EVERY=${FORK_EVERY:-4}
export SYS_TOK=${SYS_TOK:-3000} FIRST_TOK=${FIRST_TOK:-600} TOOL_TOK=${TOOL_TOK:-700}
export MAX_TOKENS=${MAX_TOKENS:-48}

cargo build --release -p tang-llm --features cuda --bin tang-llm 2>&1 | tail -2
LLM=${CARGO_TARGET_DIR:-target}/release/tang-llm
free -g | head -2 | tee "$OUT/mem.txt"
df -h /tmp "${COLD:-/tmp}" | tee "$OUT/disks.txt"

pid=
declare -A dirs
trap '[ -n "$pid" ] && kill $pid 2>/dev/null; for d in "${dirs[@]}"; do rm -rf "$d"; done' EXIT
if curl -sf localhost:8931/v1/models >/dev/null 2>&1; then echo "port 8931 already serving" >&2; exit 1; fi
for mode in $MODES; do
  base=${mode%+}
  if [ "$mode" != "$base" ]; then
    state=${dirs[$base]:?"$mode needs $base earlier in MODES"}
  else
    state=$(mktemp -d)
    dirs[$base]=$state
  fi
  envs=(TANG_FLASH_STATE_DIR="$state" TANG_SNAP_TIMES=1 TANG_FLASH_DISK_GB=200)
  case $base in
    off) envs+=(TANG_SERVE_CACHE=off) ;;
    s249) envs+=(TANG_SERVE_SHARE=all TANG_SERVE_SPLIT_MIN=1000000000 TANG_KV_WRITEBACK=0 TANG_SERVE_REUSE_W=0) ;;
    radix) envs+=(TANG_SERVE_SHARE=all TANG_KV_WRITETHROUGH=1) ;;
    radix-sys) envs+=(TANG_SERVE_SHARE=system TANG_KV_WRITETHROUGH=1) ;;
    radix-cold) envs+=(TANG_SERVE_SHARE=all TANG_KV_WRITETHROUGH=1 TANG_FLASH_DISK_GB=0 TANG_KV_COLD_DIR="${COLD:?}/$(basename "$state")") ;;
    *) echo "unknown mode $mode" >&2; exit 1 ;;
  esac
  log="$OUT/serve-$mode.log"
  env "${envs[@]}" "$LLM" serve "$GGUF" --port 8931 --ctx "$CTX" --batch "$BATCH" >"$log" 2>&1 &
  pid=$!
  for _ in $(seq 600); do
    curl -sf localhost:8931/v1/models >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || { tail -30 "$log"; exit 1; }
    sleep 2
  done
  LABEL=$mode OUT=$OUT URL=http://127.0.0.1:8931 python3 scripts/prefix-replay.py | tee -a "$OUT/summary.txt"
  { grep -E "sequence slots fit|cache:" "$log" || true; } | tail -2 | tee -a "$OUT/summary.txt"
  { grep -E "from (hot|cold) disk" "$log" || true; } | awk '{mb+=$(NF-6); ms+=$(NF-3); n++} END {if (n) printf "disk reads %d: %.0f MB in %.0f ms = %.2f GB/s\n", n, mb, ms, mb/ms}' | tee -a "$OUT/summary.txt"
  du -sh "$state" | sed 's/^/state dir /' | tee -a "$OUT/summary.txt"
  kill $pid; wait $pid 2>/dev/null || true; pid=
  sleep 3
done

# Raw disk bandwidth (O_DIRECT, so not the page cache): what a restore can read at.
bw() {
  local f=$1/bw.$$
  dd if=/dev/zero of="$f" bs=16M count=128 oflag=direct status=none
  echo "$1: $(dd if="$f" of=/dev/null bs=16M iflag=direct 2>&1 | tail -1)" | tee -a "$OUT/summary.txt"
  rm -f "$f"
}
bw /tmp
[ -n "${COLD:-}" ] && { mkdir -p "$COLD"; bw "$COLD"; }
true
