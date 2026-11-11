#!/usr/bin/env bash
# Remote half of scripts/glm5-bench.sh — runs ON the GPU box from the md5-verified tree:
#   bash $R/src/scripts/glm5-bench-remote.sh  (env: R G5 FLAGS LABEL COMMIT N PROMPT)
# Writes $R/out/<LABEL>/{gate,bench}.log, $R/out/<LABEL>/summary and $R/out/<LABEL>/DONE.
set -u
O=$R/out/$LABEL; rm -rf "$O"; mkdir -p "$O"
fin() { echo "$1" > "$O/summary"; echo "$1"; touch "$O/DONE"; exit 0; }
BIN=$R/src/target/release/tang-llm
ORACLE="785 1196 6801 264 6386 38038 911 279"
read -r -a FE <<< "$FLAGS"
unset TANG_GLM5_GPU_FFN TANG_GLM5_GPU_FFN_LAYERS TANG_GLM5_GPU TANG_GLM5_TRACE
exec 8> "$HOME/.gpu.lock"; flock 8   # the launcher also holds /root/gpu.lock
# 1. ids gate
env "${FE[@]+"${FE[@]}"}" timeout 3600 "$BIN" flash-generate "$G5" --prompt 'Write a haiku about the sea.' \
  -n 8 --temp 0 > "$O/gate.log" 2>&1 < /dev/null
IDS=$(grep -a '^ids: ' "$O/gate.log" | tail -1 | cut -c6-)
[ "$IDS" = "$ORACLE" ] || fin "bench $COMMIT $FLAGS: ids MISMATCH (got '${IDS:-none}', want '$ORACLE'; $O/gate.log)"
# 2. timed decode, traced
env "${FE[@]+"${FE[@]}"}" TANG_GLM5_TRACE=1 timeout 5400 "$BIN" flash-generate "$G5" --prompt "$PROMPT" \
  -n "$N" --temp 0 > "$O/bench.log" 2>&1 < /dev/null
RC=$?
PL=$(grep -a '^prompt: ' "$O/bench.log" | tail -1)
PT=$(sed -nE 's/.*\(([0-9.]+) tok\/s\).*/\1/p' <<< "$PL")
PN=$(sed -nE 's/^prompt: ([0-9]+) tokens.*/\1/p' <<< "$PL")
DT=$(grep -a '^decode: ' "$O/bench.log" | tail -1 | sed -nE 's/.*\(([0-9.]+) tok\/s\).*/\1/p')
# per-stage ms/token: top-level (unindented) rows of the decode table, wall ms column;
# mhc = mhc_pre.attn + mhc_post.attn + mhc_pre.ffn + mhc_post.ffn
ST=$(awk '/^-- decode, per token/{d=1;next} d&&/^(-- |kernels by)/{d=0} d&&/^[a-z]/&&$1!="stage"{
  s=$1; sub(/[._].*/,"",s); w[s]+=$3} END{printf "embed %.2f ms kda %.2f ms dsa %.2f ms mhc %.2f ms ffn %.2f ms head %.2f ms",
  w["embed"],w["kda"],w["dsa"],w["mhc"],w["ffn"],w["head"]}' "$O/bench.log")
[ "$RC" = 0 ] || fin "bench $COMMIT $FLAGS: ids OK, timed run rc=$RC ($O/bench.log)"
fin "bench $COMMIT ${FLAGS:-naive}: ids OK, prefill $PT tok/s ($PN tok), decode $DT tok/s, $ST"
