#!/usr/bin/env bash
# EP tests (KL-gated) with TCX off and on, plus a jitter control (same build, K-split S
# changed) whose KL is the bar TCX must stay under. Needs >= 2 GPUs for the NCCL test
# (~20 min).
source "$(dirname "$0")/common.sh"
cargo test -p tang-llm --features cuda,ep --lib --no-run 2>&1 | tail -1
BIN=$(ls -t "$CARGO_TARGET_DIR"/debug/deps/tang_llm-* | grep -v '\.d$' | head -1)
for v in 0 1; do
  DS_MOE_TCX=$v "$BIN" --test-threads=1 --nocapture deepseek::ep:: 2>&1 | tee "$LOG/ep_tcx$v.txt" | grep -E "KL|test result"
done
DS_MOE_KS_S=2 "$BIN" --test-threads=1 --nocapture deepseek::ep:: 2>&1 | tee "$LOG/ep_jitter.txt" | grep -E "KL|test result"
