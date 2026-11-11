#!/usr/bin/env bash
# Remote half of scripts/glm5-b300-bench.sh — runs ON the GPU box, detached, from the
# md5-verified tree in $R/src:   cd $R && bash src/scripts/glm5-b300-bench-remote.sh in/bench.env
#
# bench.env (written by the client): TREE NF TGZ_MD5 R G5 ORACLE PROMPT PROMPT_IDS
# N_SINGLE N_SLOTS SLOTS XENV.
#
# 1. cuda build (CARGO_TARGET_DIR=$R/target, outside src) IN PARALLEL with the weights
#    fetch (only when G5 is a 6-shard *-00001-of-00006.gguf set with shards missing).
#    Binary md5 before / after: a failed build aborts; an UNCHANGED binary aborts unless
#    the build inputs (the tree minus scripts/docs/tests/*.md) are the same as the last
#    build's ($R/target/bench-inputs.md5), i.e. an identical rebuild of identical sources.
# 2. ORACLE=cpu: the oracle ids from a CPU run (TANG_GLM5_GPU=0, -n 8) — small models only.
# 3. under flock ~/.cache/tang-gpu.lock, all-GPU (TANG_GLM5_MOE=fast GEMV8 KDA_PF2
#    MHC_FUSED GPU_HEAD MLA_RESIDENT; TANG_GLM5_GPU_FFN unset), plus XENV:
#      single stream -n N_SINGLE, TANG_GLM5_TRACE=1: ids gate (first 8 = ORACLE) + tok/s;
#      --slots N --diverse -n N_SLOTS for N in SLOTS: aggregate tok/s, slot 0 = ORACLE prefix.
# 4. $R/summary.md (one table) and $R/status (timestamped lines; last line DONE, also on
#    abort). Logs in $R/logs.
set -u
ENVF=${1:?usage: glm5-b300-bench-remote.sh in/bench.env}
# shellcheck disable=SC1090
source "$ENVF"
cd "$R" || exit 1
S=$R/status L=$R/logs BIN=$R/target/release/tang-llm
FAST="TANG_GLM5_MOE=fast TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_MHC_FUSED=1 TANG_GLM5_GPU_HEAD=1 TANG_GLM5_MLA_RESIDENT=1"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$S"; }
ABORTED=
abort() { ABORTED=$*; st "ABORT: $*"; exit 1; }
ROWS=()
row() { ROWS+=("| $1 | $2 | $3 | $4 |"); }
summary() {
  {
    echo "# GLM-5.3-Flash bench — $(hostname), $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo
    echo "- tree \`$TREE\`, binary md5 \`${BIN_MD5:-none}\`"
    echo "- gpu: ${GPU_NAME:-?}"
    echo "- model: \`$G5\`"
    if [ -n "$PROMPT_IDS" ]; then echo "- prompt ids: \`$PROMPT_IDS\`"; else echo "- prompt: \"$PROMPT\""; fi
    echo "- oracle (first 8): \`${ORACLE_IDS:-$ORACLE}\`"
    echo "- env: \`$FAST${XENV:+ $XENV}\`"
    case " $XENV " in *" TANG_GLM5_GPU_FFN=0 "*)
      echo "- **NOTE: GLM5_ENV sets TANG_GLM5_GPU_FFN=0 — the FFNs run on the CPU. This is NOT the all-GPU number.**" ;;
    esac
    [ -n "$ABORTED" ] && echo "- **ABORTED: $ABORTED**"
    echo
    echo "| config | ids ok? | tok/s | detail |"
    echo "|---|---|---|---|"
    for r in "${ROWS[@]+"${ROWS[@]}"}"; do echo "$r"; done
    echo
    echo "tok/s: single = decode tok/s (the -n line, traced); slots = aggregate decode tok/s over all slots."
    echo "ids ok?: single = its first 8 ids vs the oracle; slots = slot 0's ids vs the oracle prefix."
  } > "$R/summary.md"
}
trap 'summary; echo DONE >> "$S"' EXIT

exec 9> "$R/run.lock"
flock -n 9 || abort "another bench run holds $R/run.lock"
rm -rf "$L"; mkdir -p "$L" "$HOME/.cache"
st "start: tree $TREE, $(nproc) cpus, model $G5"

export PATH=$HOME/.cargo/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd';' -)
[ -n "$GPU_NAME" ] || abort "nvidia-smi: no GPU"
st "gpu: $GPU_NAME"
[ -x /usr/local/cuda/bin/nvcc ] || abort "no /usr/local/cuda/bin/nvcc"
if ! command -v cargo > /dev/null; then
  st "no cargo — installing rustup (stable, minimal)"
  curl -sSf https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal --default-toolchain stable > "$L/rustup.log" 2>&1 \
    || abort "rustup install failed (logs/rustup.log)"
  export PATH=$HOME/.cargo/bin:$PATH
fi

# 1a. weights: a 6-shard set is fetched if any shard is missing; anything else must exist
GD=$(dirname "$G5")
shards() { ls "$GD"/*-of-00006.gguf 2>/dev/null | wc -l; }
FETCH_PID=
case "$G5" in
  *GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf)
    if [ "$(shards)" != 6 ]; then
      st "weights incomplete ($(shards)/6) — fetching unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL into $(dirname "$GD") (logs/download.log), in parallel with the build"
      (
        command -v hf > /dev/null || pip install -q --break-system-packages -U huggingface_hub \
          || pip3 install -q --break-system-packages -U huggingface_hub
        hf download unsloth/GLM-5.3-Flash-GGUF --include '*UD-Q4_K_XL*' --local-dir "$(dirname "$GD")"
      ) > "$L/download.log" 2>&1 < /dev/null &
      FETCH_PID=$!
    else
      st "weights present: 6/6 shards in $GD"
    fi ;;
  *) [ -f "$G5" ] || abort "model $G5 missing (only the default 6-shard UD-Q4_K_XL set is fetched)"
     st "weights present: $G5" ;;
esac

# 1b. build. touch + rm the binary: cargo must recompile tang-llm and relink from THIS tree
INPUTS=$(grep -vE '  (scripts|docs)/|/tests/|\.md$' in/manifest | md5sum | cut -c1-32)
BEFORE=$( [ -f "$BIN" ] && md5sum "$BIN" | cut -c1-32 || echo none)
LAST_INPUTS=$(cat "$R/target/bench-inputs.md5" 2>/dev/null || echo none)
st "build: binary before $BEFORE (logs/build.log)"
touch src/crates/tang-llm/src/lib.rs
rm -f "$BIN"
(cd src && CARGO_TARGET_DIR=$R/target cargo build --release -p tang-llm --bin tang-llm \
  --no-default-features --features cuda) > "$L/build.log" 2>&1 < /dev/null
BRC=$?
if [ -n "$FETCH_PID" ] && [ "$BRC" != 0 ]; then kill "$FETCH_PID" 2> /dev/null; fi
[ "$BRC" = 0 ] || abort "cuda build failed rc=$BRC (logs/build.log: $(grep -aE '^error' "$L/build.log" | head -1))"
[ -x "$BIN" ] || abort "build rc=0 but no $BIN"
BIN_MD5=$(md5sum "$BIN" | cut -c1-32)
if [ "$BIN_MD5" = "$BEFORE" ]; then
  [ "$INPUTS" = "$LAST_INPUTS" ] || abort "binary UNCHANGED ($BIN_MD5) but the build inputs changed since the last build — stale build"
  st "build ok: binary $BIN_MD5 (identical rebuild; build inputs unchanged since the last build)"
else
  st "build ok: binary $BEFORE -> $BIN_MD5"
fi
echo "$INPUTS" > "$R/target/bench-inputs.md5"

if [ -n "$FETCH_PID" ]; then
  st "waiting for the weights fetch"
  wait "$FETCH_PID"; FRC=$?
  [ "$FRC" = 0 ] && [ "$(shards)" = 6 ] || abort "weights fetch rc=$FRC, $(shards)/6 shards (logs/download.log)"
  st "weights fetched: 6/6 shards, $(du -sh "$GD" | cut -f1)"
fi

# prompt args
if [ -n "$PROMPT_IDS" ]; then PARGS=(--prompt-ids "$PROMPT_IDS"); else PARGS=(--prompt "$PROMPT"); fi
# word-split XENV into env assignments (values without spaces)
read -r -a XE <<< "$XENV"
first() { tr -s ' ' '\n' <<< "$1" | grep . | head -n "$2" | paste -sd' ' -; }
nw() { wc -w <<< "$1" | tr -d ' '; }
# ok if got's first k ids = the oracle's, k = min(8, oracle length, want)
idsok() {
  local got=$1 want=$2 k
  k=$(nw "$ORACLE_IDS"); [ "$k" -gt 8 ] && k=8; [ "$want" -lt "$k" ] && k=$want
  [ "$k" -gt 0 ] || { echo "n/a"; return; }
  if [ "$(first "$got" "$k")" = "$(first "$ORACLE_IDS" "$k")" ]; then echo "yes ($k/$k)"; else echo "**NO** (got \`$(first "$got" "$k")\`)"; fi
}

# 2. the oracle
if [ "$ORACLE" = cpu ]; then
  st "oracle: CPU run (TANG_GLM5_GPU=0, -n 8; logs/oracle-cpu.log)"
  env "${XE[@]+"${XE[@]}"}" TANG_GLM5_GPU=0 timeout 3600 "$BIN" flash-generate "$G5" "${PARGS[@]}" -n 8 --temp 0 \
    --out "$L/oracle.ids" > "$L/oracle-cpu.log" 2>&1 < /dev/null
  ORC=$?
  [ "$ORC" = 0 ] || abort "CPU oracle run rc=$ORC (logs/oracle-cpu.log)"
  ORACLE_IDS=$(first "$(cat "$L/oracle.ids")" 8)
  st "oracle (cpu): $ORACLE_IDS"
else
  ORACLE_IDS=$(first "$ORACLE" 8)
fi

# 3. all-GPU runs, one lock for the whole set
unset TANG_GLM5_GPU_FFN TANG_GLM5_GPU TANG_GLM5_TRACE
# shellcheck disable=SC2086,SC2163
export $FAST
exec 8> "$HOME/.cache/tang-gpu.lock"
st "waiting for $HOME/.cache/tang-gpu.lock"
flock 8
st "gpu lock held; gpu memory used, total: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | paste -sd';' -)"

st "single: -n $N_SINGLE, TANG_GLM5_TRACE=1 (logs/single.log)"
env "${XE[@]+"${XE[@]}"}" TANG_GLM5_TRACE=1 timeout 5400 "$BIN" flash-generate "$G5" "${PARGS[@]}" -n "$N_SINGLE" --temp 0 \
  > "$L/single.log" 2>&1 < /dev/null
RC=$?
IDS=$(grep -a '^ids: ' "$L/single.log" | tail -1 | cut -c6-)
TPS=$(grep -a '^decode: ' "$L/single.log" | tail -1 | sed -E 's/.*\(([0-9.]+) tok\/s\).*/\1/')
PRE=$(grep -a '^prompt: ' "$L/single.log" | tail -1 | sed -E 's/^prompt: //')
OK=$( [ -n "$IDS" ] && idsok "$IDS" "$N_SINGLE" || echo "**NO** (no ids)")
row "single, -n $N_SINGLE, traced" "$OK" "${TPS:-—}" "rc $RC; $PRE"
st "single rc=$RC ids ok? $OK, decode ${TPS:-?} tok/s; first ids: $(first "$IDS" 8)"

for N in $SLOTS; do
  st "slots $N: --diverse -n $N_SLOTS (logs/slots$N.log)"
  env "${XE[@]+"${XE[@]}"}" timeout 10800 "$BIN" flash-generate "$G5" "${PARGS[@]}" -n "$N_SLOTS" --temp 0 --slots "$N" --diverse \
    > "$L/slots$N.log" 2>&1 < /dev/null
  RC=$?
  IDS=$(grep -a '^ids slot 0: ' "$L/slots$N.log" | tail -1 | cut -c13-)
  LINE=$(grep -a "^slots $N " "$L/slots$N.log" | tail -1)
  TPS=$(sed -nE 's/.*aggregate ([0-9.]+) tok\/s.*/\1/p' <<< "$LINE")
  PS=$(sed -nE 's/.*per stream ([0-9.]+) tok\/s.*/\1/p' <<< "$LINE")
  MS=$(sed -nE 's/.*per-step wall ([0-9.]+) ms median.*/\1/p' <<< "$LINE")
  DIV=$(grep -a '^diverse prompts: ' "$L/slots$N.log" | tail -1 | sed -E 's/^diverse prompts: //')
  OK=$( [ -n "$IDS" ] && idsok "$IDS" "$N_SLOTS" || echo "**NO** (no ids)")
  row "--slots $N --diverse, -n $N_SLOTS" "$OK" "${TPS:-—}" "rc $RC; per stream ${PS:-?} tok/s, step ${MS:-?} ms median; $DIV"
  st "slots $N rc=$RC slot-0 ids ok? $OK, aggregate ${TPS:-?} tok/s (per stream ${PS:-?})"
done
st "finished"
