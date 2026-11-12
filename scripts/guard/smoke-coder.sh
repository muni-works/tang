#!/usr/bin/env bash
# GPU smoke of the derived coder-next step (runs on a runner pod as a job; MODEL_DIR set
# by the runner): one chunked prefill and one decode step on the real UD-Q4_K_XL GGUF.
# Writes $OUT/summary.txt; exits nonzero on a failed build or run.
set -uo pipefail
OUT=${OUT:-/tmp/smoke-coder}; mkdir -p "$OUT"
GGUF=${MODEL_DIR:?MODEL_DIR}/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
CHUNK=${CHUNK:-16} CTX=${CTX:-512}
sum() { echo "$*" | tee -a "$OUT/summary.txt"; }
: >"$OUT/summary.txt"
sum "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
sum "commit: $(git rev-parse --short HEAD)"
if ! cargo build --release -p tang-bench --features cuda,gguf --example coder_derived \
    >"$OUT/build.txt" 2>&1; then
  sum "build: FAIL"; tail -30 "$OUT/build.txt"; exit 1
fi
DER=target/release/examples/coder_derived
# 2 chunks of prompt (any ids below the vocab)
seq 1000 37 $((1000 + 37 * (2 * CHUNK - 1))) | tr '\n' ' ' >"$OUT/prompt.ids"
rc=0
if "$DER" "$GGUF" --ids-file "$OUT/prompt.ids" --chunk "$CHUNK" --ctx "$CTX" --no-graph \
    >"$OUT/prefill.txt" 2>&1; then
  sum "chunked prefill (k=$CHUNK): ok $(grep -iE 'prefill|tok/s' "$OUT/prefill.txt" | tail -1)"
else
  sum "chunked prefill (k=$CHUNK): FAIL"; tail -30 "$OUT/prefill.txt"; rc=1
fi
if "$DER" "$GGUF" --ids-file "$OUT/prompt.ids" -n 1 --ctx "$CTX" --out "$OUT/ids.txt" \
    >"$OUT/decode.txt" 2>&1 && [ -s "$OUT/ids.txt" ]; then
  sum "decode (1 step): ok, token $(tr -s ' \n' ' ' <"$OUT/ids.txt")"
else
  sum "decode (1 step): FAIL"; tail -30 "$OUT/decode.txt"; rc=1
fi
exit $rc
