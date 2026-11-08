#!/bin/bash
# Per-kernel durations and start times (CUDA graph nodes included) when hardware counters
# are not available (containers without NVreg_RestrictProfilingToAdminUsers=0, e.g. RunPod):
#
#   scripts/ncu/nsys_kernels.sh OUT.csv [nsys profile options] -- command args...
#
# `--capture-range=cudaProfilerApi` with TANG_PROFILE_STEPS takes one decode step.
set -euo pipefail
out=$1; shift
opts=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
shift
NSYS=${NSYS:-nsys}
$NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$out.prof" "${opts[@]}" "$@" > "$out.raw" 2>&1
$NSYS stats -q --report cuda_gpu_trace --format csv "$out.prof.nsys-rep" 2>/dev/null \
  | awk -F, 'NR==1 || ($NF !~ /^\[CUDA mem/)' > "$out"
echo "$(($(wc -l < "$out") - 1)) kernel launches -> $out" >&2
