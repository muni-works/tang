#!/usr/bin/env bash
# Kernel gates on one GPU with the tiny fixture (any sm_80+ card; ~10 min incl. build).
# TCX vs K-split must match bitwise; the whole deepseek GPU suite must pass with TCX on.
source "$(dirname "$0")/common.sh"
cargo test -p tang-llm --features cuda --lib --no-run 2>&1 | tail -1
BIN=$(ls -t "$CARGO_TARGET_DIR"/debug/deps/tang_llm-* | grep -v '\.d$' | head -1)
"$BIN" --test-threads=1 deepseek::engine::tests::gpu_tiny_moe_tcx_matches_ks | tee "$LOG/tcx_bitwise.txt"
DS_MOE_TCX=0 "$BIN" --test-threads=1 deepseek:: | tee "$LOG/suite_tcx0.txt" | grep "test result"
DS_MOE_TCX=1 "$BIN" --test-threads=1 deepseek:: | tee "$LOG/suite_tcx1.txt" | grep "test result"
