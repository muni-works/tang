#!/usr/bin/env bash
# Real weights, 8 GPUs EP: decode tok/s TCX off vs on (deepseek-profile, 200 tokens, spec on),
# then served greedy/sampled through deepseek-serve. ~45 min with weights already on disk.
source "$(dirname "$0")/common.sh"
[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
cargo build --release -p tang-llm --features cuda,ep --bin tang-llm 2>&1 | tail -1
TL="$CARGO_TARGET_DIR/release/tang-llm"
DEV=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
for v in 0 1; do
  for run in 1 2; do
    DS_MOE_TCX=$v "$TL" deepseek-profile "$CKPT" --devices "$DEV" --steps 200 2>&1 | tee "$LOG/profile_tcx${v}_$run.txt" | grep -E "tok/s|grouped_(gu|down)"
  done
done
