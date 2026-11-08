#!/bin/bash
# The fraction-of-light sweep: every workload through ncu_kernels.sh, then table.py.
#
#   scripts/ncu/run_all.sh OUTDIR BW_BYTES_PER_S [small|all]
#
# small: derived MoE layer (T = 1, 8, 48), the engine's small ops (Q8 GEMV etc., derived and
#        hand), the GDN chunked scan, the fused attention backward check.
# all:   + derived flash attention fwd/bwd (hopper_attention; needs torch for the inputs) and
#        the full Coder-Next decode step per kernel, hand and derived MoE, at 1 / 8 / 48
#        streams (needs MODEL=path.gguf and PROMPT=file).
# Env: NCU (ncu binary), STUBS (cudarc_stubs.so), TGT (cargo target dir), SUDO=sudo.
# Exits non-zero at the first failure.
set -euo pipefail
out=$1; bw=$2; set_=${3:-small}
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
TGT=${TGT:-$root/target}; SUDO=${SUDO:-}
mkdir -p "$out"
prof() { # name [ncu opts] -- cmd...
  local name=$1; shift
  echo "== $name" >&2
  $SUDO env NCU="${NCU:-ncu}" STUBS="${STUBS:-}" ONCE="${ONCE:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    "$here/ncu_kernels.sh" "$out/$name.csv" "$@" 2> "$out/$name.err" || { tail -5 "$out/$name.err"; return 1; }
  python3 "$here/table.py" "$out/$name.csv" --bw "$bw" --title "$name" --json "$out/$name.json" ${STEP_MS:+--step-ms $STEP_MS} > "$out/$name.md"
  tail -2 "$out/$name.md" >&2
}
EX=$TGT/release/examples
for t in 1 8 48; do ONCE=1 prof moe_t$t -k regex:dop -- "$EX/coder_derived_moe" $t; done
ONCE=1 prof ops -- "$EX/coder_derived_ops" 1 8
prof gdn_scan -- "$EX/scan_bench" gdn 32 4096
prof attn_bwd --launch-count 400 -- env CASES=8x4096x128x0 "$EX/attn_bwd_check"
[ "$set_" = all ] || exit 0

# derived flash attention fwd + bwd (bf16, N 4096, D 128)
python3 "$root/scripts/hopper_attention.py" gen "$out/attn_in" 4096 128 0
prof flash_attn --launch-count 400 -- "$EX/hopper_attention" "$out/attn_in"

# full decode step, one step per config inside cuProfilerStart / Stop
: "${MODEL:?MODEL=path.gguf}" "${PROMPT:?PROMPT=file}"
BIN=$TGT/release/tang-llm
for b in 1 8 48; do
  for v in hand der; do
    envs=(TANG_BATCH_N=$b TANG_BATCH_ONLY=$b TANG_PROFILE_STEPS=1)
    [ $v = der ] && envs+=(TANG_CODER_DERIVED=moe TANG_DMOE_MIN_ROWS=1)
    # batch-check exits 1 on a tolerance FAIL; the profile still counts. The wall time comes
    # from the same run's timed steps, which run before the profiled window.
    prof step_${v}_b$b --profile-from-start off -- bash -c '"$@"; exit 0' _ env "${envs[@]}" "$BIN" flash-batch-check "$MODEL" --prompt-file "$PROMPT" -n 16
    STEP_MS=$(grep -oE "^batch $b: [0-9.]+ ms a step" "$out/step_${v}_b$b.csv.raw" | grep -oE "[0-9.]+ ms" | grep -oE "[0-9.]+" | head -1)
    [ -n "$STEP_MS" ] || { echo "no step time for $v b$b" >&2; exit 1; }
    python3 "$here/table.py" "$out/step_${v}_b$b.csv" --bw "$bw" --title "step_${v}_b$b" --step-ms "$STEP_MS" --json "$out/step_${v}_b$b.json" > "$out/step_${v}_b$b.md"
    tail -2 "$out/step_${v}_b$b.md" >&2
  done
done
