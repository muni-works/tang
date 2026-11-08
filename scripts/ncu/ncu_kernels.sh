#!/bin/bash
# Per-kernel hardware counters for any CUDA command, as CSV for table.py.
#
#   scripts/ncu/ncu_kernels.sh OUT.csv [ncu options] -- command args...
#
# ncu options pass through (e.g. `-k regex:dop|cx_` to filter kernel names,
# `--launch-skip 2000 --launch-count 800` to take one decode step after warm-up).
# Metrics: DRAM bytes read / written, kernel duration, SM and DRAM throughput
# (% of peak), launch shape. Needs profiling permission: root, or
# NVreg_RestrictProfilingToAdminUsers=0 (`NCU=sudo ncu` on a shared box).
set -euo pipefail
out=$1; shift
opts=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "usage: $0 OUT.csv [ncu opts] -- cmd..." >&2; exit 2; }
shift
NCU=${NCU:-ncu}
# STUBS=path/to/cudarc_stubs.so: preload stubs for symbols cudarc expects (see cudarc_stubs.c)
pre=()
[ -n "${STUBS:-}" ] && pre=(env LD_PRELOAD="$STUBS")
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
M=$M,sm__throughput.avg.pct_of_peak_sustained_elapsed
M=$M,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
M=$M,dram__throughput.avg.pct_of_peak_sustained_elapsed
M=$M,launch__grid_size,launch__block_size
# --cache-control none / --clock-control none: measure as the program runs (warm L2, boost
# clocks), not ncu's flushed-cache, base-clock defaults
$NCU --csv --print-units base --metrics "$M" --cache-control none --clock-control none \
  --target-processes all "${opts[@]}" "${pre[@]}" "$@" > "$out.raw"
# keep the CSV rows only (the program's own stdout is interleaved)
grep '^"' "$out.raw" > "$out"
echo "$(($(wc -l < "$out") - 1)) metric rows -> $out" >&2
