#!/bin/bash
# The run_all.sh workloads through nsys (kernel durations and gaps only), for GPUs whose
# counters ncu cannot read. Tables: nsys_table.py, bytes joined from a run_all.sh JSON.
#
#   scripts/ncu/run_nsys.sh OUTDIR        (env: NSYS, TGT, MODEL, PROMPT)
set -euo pipefail
out=$1; here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
TGT=${TGT:-$root/target}; EX=$TGT/release/examples; mkdir -p "$out"
p() { local n=$1; shift; echo "== $n" >&2; "$here/nsys_kernels.sh" "$out/$n.csv" "$@" 2> "$out/$n.err" || { tail -5 "$out/$n.err" "$out/$n.csv.raw" >&2; return 1; }; tail -1 "$out/$n.err" >&2; }
for t in 1 8 48; do p moe_t$t -- env ONCE=1 "$EX/coder_derived_moe" $t; done
for t in 1 8; do p ops_t$t -- env ONCE=1 "$EX/coder_derived_ops" $t; done
p gdn_scan -- "$EX/scan_bench" gdn 32 4096
p attn_bwd -- env CASES=8x4096x128x0 "$EX/attn_bwd_check"
python3 "$root/scripts/hopper_attention.py" gen "$out/attn_in" 4096 128 0 > "$out/attn_gen.log" 2>&1
p flash_attn -- "$EX/hopper_attention" "$out/attn_in"
: "${MODEL:?}" "${PROMPT:?}"
for b in 1 8 48; do
  for v in hand der; do
    envs=(TANG_BATCH_N=$b TANG_BATCH_ONLY=$b TANG_PROFILE_STEPS=1)
    [ $v = der ] && envs+=(TANG_CODER_DERIVED=moe TANG_DMOE_MIN_ROWS=1)
    p step_${v}_b$b --capture-range=cudaProfilerApi --capture-range-end=stop -- bash -c '"$@"; exit 0' _ env "${envs[@]}" "$TGT/release/tang-llm" flash-batch-check "$MODEL" --prompt-file "$PROMPT" -n 16
    grep -oE "^batch $b: [0-9.]+ ms a step" "$out/step_${v}_b$b.csv.raw" | head -1 > "$out/step_${v}_b$b.wall"
    [ -s "$out/step_${v}_b$b.wall" ] || { echo "no step time $v b$b" >&2; exit 1; }
  done
done
