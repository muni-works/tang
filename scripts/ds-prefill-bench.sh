#!/usr/bin/env bash
# DS-V4.1-Flash 2-layer (0, 2) native prefill timing on one GPU, scaled x20 to 40 layers.
#   CONFIGS="T:k:group:vendor ..."  (default: baseline + grouped at 4k and 32k)
# Each config times the k-token step at the first and last chunk position and
# reports T / (chunks * mean step) as prefill tok/s for 2 layers and for 40.
set -uo pipefail
DS41_DIR=${DS41_DIR:-/root/ds41sol}
CONFIGS=${CONFIGS:-"4096:256:0:0 4096:4096:0:0 4096:4096:64:0 32768:4096:0:0 32768:4096:64:0"}
OUT=${OUT:-/tmp/ds-prefill}; mkdir -p "$OUT"
cargo build --release -p tang-bench --features cuda --example ds_real_gpu 2>&1 | tail -3
bin=${CARGO_TARGET_DIR:-target}/release/examples/ds_real_gpu
: > "$OUT/summary.txt"
for c in $CONFIGS; do
  IFS=: read -r T k g v <<<"$c"
  ms=()
  for pos in 0 $((T - k)); do
    [ "$pos" -lt 3 ] && pos=3
    line=$(DS41_DIR=$DS41_DIR DS_NATIVE=1 DS_ALL_EXPERTS=1 DS_NO_CHECK=1 DS_STEP_K=$k DS_REAL_T=$pos \
      TANG_DS_GROUP=$g TANG_VENDOR_GEMM=$([ "$v" = 1 ] && echo 1 || echo "") DS_TIME_ITERS=${ITERS:-5} \
      "$bin" 2>"$OUT/err-$T-$k-$g-$v-$pos.txt" | tail -1)
    echo "$c pos $pos: $line" | tee -a "$OUT/log.txt"
    m=$(sed -n 's/.* \([0-9.]*\) ms device.*/\1/p' <<<"$line")
    [ -n "$m" ] || { tail -5 "$OUT/err-$T-$k-$g-$v-$pos.txt"; m=nan; }
    ms+=("$m")
  done
  python3 - "$T" "$k" "$g" "$v" "${ms[@]}" >>"$OUT/summary.txt" <<'PY'
import sys, math
T,k,g,v=map(int,sys.argv[1:5]); a,b=map(float,sys.argv[5:7])
mean=(a+b)/2; chunks=T//k; tot=chunks*mean
print(f"T {T} k {k} group {g} vendor {v}: step {a:.2f}..{b:.2f} ms, 2-layer {T/tot*1e3:,.0f} tok/s, "
      f"40-layer (x20) {T/(tot*20)*1e3:,.0f} tok/s = {T/(tot*20)*1e3/47660*100:.2f}% of 47.7k SOL")
PY
done
cat "$OUT/summary.txt"
