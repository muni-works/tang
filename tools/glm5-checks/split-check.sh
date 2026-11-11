#!/bin/bash
# GLM-5.3-Flash / Qwen3-Coder-Next: W6 split checks on a 2×H200 pod.
#
# What this proves, in order:
#   1. eager (no CUDA graphs) decode on this branch equals eager decode on origin/main,
#      token-for-token — the 1-GPU hot path is untouched (the merge blocker).
#   2. captured-graphs decode on this branch equals eager decode on this branch —
#      the graph capture change (per-stage capture) didn't alter the captured kernels.
#   3. TANG_GPUS=2 decode produces tokens, and its tokens equal the 1-GPU run's.
#      SPLIT_AT=0 (all layers on GPU 1) is the strongest wiring test: every boundary
#      copy, cross-device norm read and stage-1 launch is exercised, with GPU 0 reduced
#      to embedding + head. SPLIT_AT=<half> is the balanced layout for perf.
#   4. (optional) bitwise logits via flash-parity --save on both engines, diffed.
# Everything is greedy (--temp 0) with fixed prompt ids, so runs are byte-comparable.
#
# Usage: MODEL=<first gguf shard> [SPLIT_AT=0] [N=64] ./tools/glm5-checks/split-check.sh
#   MODEL: e.g. /root/Qwen3-Coder-Next-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00003.gguf
# Exits nonzero on the first failed check; prints PASSED per check.
set -euo pipefail

MODEL="${MODEL:?set MODEL to the first GGUF shard}"
SPLIT_AT="${SPLIT_AT:-0}"      # 0 = all layers on GPU 1 (max boundary coverage)
N="${N:-64}"
IDS="--prompt-ids 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 \
1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 \
1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 \
1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 \
1060 1061 1062 1063"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAILED: $1" >&2; exit 1; }

echo "== build (branch)"
cargo build --release -p tang-llm --no-default-features --features cuda
BIN=./target/release/tang-llm

echo "== build (origin/main)"
git worktree add --detach "$TMP/main" origin/main >/dev/null 2>&1 \
    || fail "cannot check out origin/main into a worktree"
(cd "$TMP/main" && cargo build --release -p tang-llm --no-default-features --features cuda) \
    || fail "origin/main does not build"
MAIN=$TMP/main/target/release/tang-llm

gen() { # gen <out-file> <binary> [env k=v ...]
    local out=$1 bin=$2; shift 2
    env "$@" $bin flash-generate "$MODEL" $IDS -n "$N" --temp 0 --draft none \
        --out "$out" >/dev/null 2>"$TMP/err.log" \
        || fail "decode crashed: $(tail -3 "$TMP/err.log")"
}

echo "== 1. this branch (eager, TANG_GPUS=1) vs origin/main (eager)"
gen "$TMP/main.ids" "$MAIN" TANG_GPUS=1 --eager
gen "$TMP/branch.ids" "$BIN" TANG_GPUS=1 --eager
if cmp -s "$TMP/main.ids" "$TMP/branch.ids"; then
    echo "  PASSED: 1-GPU eager tokens identical to origin/main"
else
    diff <(head -20 "$TMP/main.ids") <(head -20 "$TMP/branch.ids") | head -4 >&2 || true
    fail "1-GPU eager tokens differ from origin/main"
fi

echo "== 2. this branch graphs vs eager (same engine)"
gen "$TMP/branch-graphs.ids" "$BIN" TANG_GPUS=1
if cmp -s "$TMP/branch.ids" "$TMP/branch-graphs.ids"; then
    echo "  PASSED: graphs == eager on this branch"
else
    fail "graphs differ from eager on this branch"
fi

echo "== 3. TANG_GPUS=2 (split at layer $SPLIT_AT)"
gen "$TMP/split.ids" "$BIN" TANG_GPUS=2 TANG_SPLIT_AT=$SPLIT_AT
if cmp -s "$TMP/branch-graphs.ids" "$TMP/split.ids"; then
    echo "  PASSED: TANG_GPUS=2 tokens == 1-GPU tokens (split at $SPLIT_AT)"
else
    diff <(head -20 "$TMP/branch-graphs.ids") <(head -20 "$TMP/split.ids") | head -4 >&2 || true
    fail "2-GPU tokens differ from 1-GPU (wiring bug)"
fi

echo "== 4. (optional) bitwise logits: flash-parity --save on both, diffed"
if env TANG_GPUS=1 $BIN flash-parity "$MODEL" $IDS --save "$TMP/one.jsonl" >/dev/null 2>&1 && \
   env TANG_GPUS=2 TANG_SPLIT_AT=$SPLIT_AT $BIN flash-parity "$MODEL" $IDS --save "$TMP/two.jsonl" >/dev/null 2>&1; then
    if cmp -s "$TMP/one.jsonl" "$TMP/two.jsonl"; then
        echo "  PASSED: logits bitwise equal (top-40 logprobs at every prompt position)"
    else
        diff "$TMP/one.jsonl" "$TMP/two.jsonl" | head -4 >&2 || true
        fail "logits differ"
    fi
else
    echo "  SKIP: parity run failed (non-fatal; the token checks above are the gate)"
fi

echo "ALL CHECKS PASSED (split at $SPLIT_AT, $N tokens)"
