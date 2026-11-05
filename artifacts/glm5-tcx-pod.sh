#!/usr/bin/env bash
# GLM-5.3-Flash TCX validation, run ON the GPU box from the tang tree (frog/glm5-tcx).
#
#   bash artifacts/glm5-tcx-pod.sh [stage ...]     stages: fetch build kernels gates single sweep prefill
#   (no args = all, in that order; `kernels` alone runs on a 1x H200 — see GPU-VALIDATION.md)
#
# Writes $OUT/summary.md and $OUT/status; the last status line is DONE or FAIL <stage>.
# Env: GLM5_DIR (weights dir, container disk; default /root/glm53 — NOT /workspace, the network
# volume caps at ~13 GB/min), OUT (default /root/glm5-tcx-out), GLM5_GPU_LOCK (flock file,
# default ~/tang-gpu.lock; GPU runs only — builds run outside it), GLM5_SLOTS ("1 4 8 16 32 64").
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$PWD
GLM5_DIR=${GLM5_DIR:-/root/glm53}
OUT=${OUT:-/root/glm5-tcx-out}
LOCK=${GLM5_GPU_LOCK:-$HOME/tang-gpu.lock}
SLOTS=${GLM5_SLOTS:-1 4 8 16 32 64}
G5=$GLM5_DIR/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
BIN=$SRC/target/release/tang-llm
ORACLE="785 1196 6801 264 6386 38038 911 279"
# the md5s NOTES records for the full ids list on the bench prompt (first 8 hex)
MD5_128=7174cdd6 MD5_512=1453cf0e
PROMPT=$(tr '\n' ' ' < "$SRC/scripts/glm5-bench-prompt.txt")
# the paused B300 flag set (NOTES "B flags"); TCX is layered on top per run
BASE=(TANG_GLM5_MOE=fast TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_MHC_FUSED=1 TANG_GLM5_MLA_RESIDENT=1 TANG_GLM5_GPU_HEAD=1)
mkdir -p "$OUT/logs"
S=$OUT/status SUM=$OUT/summary.md STAGE=init
st() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$S"; }
row() { echo "| $* |" >> "$SUM"; }
trap 'rc=$?; if [ $rc -eq 0 ]; then echo DONE >> "$S"; else echo "FAIL $STAGE (rc $rc)" >> "$S"; fi' EXIT
gpu() { flock "$LOCK" env "${BASE[@]}" "$@"; }
idmd5() { printf '%s' "$1" | md5sum | cut -c1-8; }
gen() { # gen <log> <n> [ENV=..]...: one greedy run on the bench prompt; echoes "ids|tok/s"
  local log=$1 n=$2; shift 2
  gpu "$@" timeout 5400 "$BIN" flash-generate "$G5" --prompt "$PROMPT" -n "$n" --temp 0 > "$log" 2>&1 < /dev/null
  local ids tps
  ids=$(grep -a '^ids: ' "$log" | tail -1 | cut -c6-)
  tps=$(grep -aoE '[0-9.]+ tok/s' "$log" | tail -1 | cut -d' ' -f1)
  echo "$ids|${tps:-?}"
}
: > "$SUM"; echo "# glm5-tcx validation $(git rev-parse --short HEAD) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$SUM"
STAGES=${*:-fetch build kernels gates single sweep prefill}

for STAGE in $STAGES; do case $STAGE in
fetch)
  st "fetch: hf_transfer -> $GLM5_DIR (container disk)"
  pip install -q -U "huggingface_hub[hf_transfer]" hf_transfer
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download unsloth/GLM-5.3-Flash-GGUF --include '*UD-Q4_K_XL*' --local-dir "$GLM5_DIR"
  [ "$(ls "$GLM5_DIR"/UD-Q4_K_XL/*.gguf | wc -l)" = 6 ] || { st "fetch: expected 6 shards"; exit 1; }
  st "fetch ok: $(du -sh "$GLM5_DIR" | cut -f1)" ;;
build)
  st "build (outside the GPU lock)"
  cargo build --release -p tang-llm --bin tang-llm --features cuda > "$OUT/logs/build.log" 2>&1
  cargo test --release -p tang-llm --features cuda --lib --no-run >> "$OUT/logs/build.log" 2>&1
  st "build ok: $(md5sum "$BIN" | cut -c1-8)" ;;
kernels)
  # per-layer kernel gate on the real weights, one layer resident at a time (fits 1x H200):
  # tcx-vs-g5f rel l2 per layer and T, routing sets, us/call. Bar: rel l2 <= 1e-3 vs CPU (the
  # fast-q8x2 bar), and tcx-vs-g5f within 2x of fast-q8x2-vs-naive.
  st "kernels: synthetic + real-layer moebench, TCX modes"
  gpu TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --features cuda --lib flash::glm5::moe -- --test-threads=1 \
    > "$OUT/logs/moe-synth.log" 2>&1
  gpu TANG_GLM5_REAL_GGUF="$G5" TANG_GLM5_MOE_BENCH_LAYERS=3,4,11,12,20,30,44 TANG_GLM5_MOE_BENCH_T=1,2,4,8,16,32,64 \
    cargo test --release -p tang-llm --features cuda --lib real_layers_fast_matches_cpu -- --nocapture --test-threads=1 \
    > "$OUT/logs/moebench.log" 2>&1
  echo; echo "## kernels" >> "$SUM"; grep -aE 'tcx-vs-g5f|tcx-q8x2 rel|fast-q8x2 rel' "$OUT/logs/moebench.log" | sed 's/^/    /' >> "$SUM"
  st "kernels ok" ;;
gates)
  # ids gates: TCX forced on at every window (MIN_T=1) must keep the oracle; md5 of the full list
  # is reported against NOTES' (tolerance path: a later-token flip is a finding, not a fail)
  st "gates: oracle ids, plain vs TCX"
  echo "## gates" >> "$SUM"; row "config | n | first 8 = oracle | md5 (want) | tok/s"; row "---|---|---|---|---"
  for n in 128 512; do
    want=$([ $n = 128 ] && echo $MD5_128 || echo $MD5_512)
    for cfg in plain tcx; do
      xe=(); [ $cfg = tcx ] && xe=(TANG_GLM5_TCX=1 TANG_GLM5_TCX_MIN_T=1)
      r=$(gen "$OUT/logs/gate-$cfg-$n.log" "$n" "${xe[@]+"${xe[@]}"}")
      ids=${r%|*}; first=$(echo "$ids" | tr -s ' ' '\n' | grep . | head -8 | paste -sd' ' -)
      ok=$([ "$first" = "$ORACLE" ] && echo yes || echo "**NO**")
      row "$cfg | $n | $ok | $(idmd5 "$ids") ($want) | ${r#*|}"
      [ "$ok" = yes ] || { st "gate $cfg -n $n: first ids $first != oracle"; exit 1; }
    done
  done
  st "gates ok" ;;
single)
  st "single stream before/after (x2 each), MTP K=1 too"
  echo "## single stream" >> "$SUM"; row "config | -n | tok/s"; row "---|---|---"
  for n in 128 512; do for rep in 1 2; do
    for cfg in plain tcx mtp1 mtp1-tcx; do
      xe=()
      case $cfg in tcx) xe=(TANG_GLM5_TCX=1) ;; mtp1) xe=(TANG_GLM5_MTP=1) ;; mtp1-tcx) xe=(TANG_GLM5_MTP=1 TANG_GLM5_TCX=1) ;; esac
      r=$(gen "$OUT/logs/single-$cfg-$n-$rep.log" "$n" "${xe[@]+"${xe[@]}"}")
      row "$cfg | $n | ${r#*|}"
    done
  done; done
  st "single ok" ;;
sweep)
  st "B-sweep: --slots N --diverse, plain vs TCX"
  echo "## B-sweep (aggregate decode tok/s)" >> "$SUM"; row "slots | plain | tcx"; row "---|---|---"
  for b in $SLOTS; do
    a=() ; for cfg in plain tcx; do
      xe=(TANG_GLM5_KDA_MAXT=64); [ $cfg = tcx ] && xe+=(TANG_GLM5_TCX=1)
      gpu "${xe[@]}" timeout 5400 "$BIN" flash-generate "$G5" --prompt "$PROMPT" -n 32 --temp 0 --slots "$b" --diverse \
        > "$OUT/logs/sweep-$cfg-$b.log" 2>&1 < /dev/null || true
      a+=("$(grep -aoE '[0-9.]+ tok/s' "$OUT/logs/sweep-$cfg-$b.log" | tail -1 | cut -d' ' -f1)")
    done
    row "$b | ${a[0]:-?} | ${a[1]:-?}"
  done
  st "sweep ok" ;;
prefill)
  st "prefill: 2k / 8k-token prompts, plain vs TCX"
  echo "## prefill" >> "$SUM"; row "tokens | plain | tcx"; row "---|---|---"
  for reps in 8 32; do
    P=$(for _ in $(seq $reps); do printf '%s ' "$PROMPT"; done)
    a=(); for cfg in plain tcx; do
      xe=(); [ $cfg = tcx ] && xe=(TANG_GLM5_TCX=1)
      gpu "${xe[@]+"${xe[@]}"}" timeout 5400 "$BIN" flash-generate "$G5" --prompt "$P" -n 4 --temp 0 \
        > "$OUT/logs/prefill-$cfg-$reps.log" 2>&1 < /dev/null
      a+=("$(grep -aiE 'prefill' "$OUT/logs/prefill-$cfg-$reps.log" | grep -aoE '[0-9.]+ tok/s' | tail -1)")
    done
    row "x$reps | ${a[0]:-?} | ${a[1]:-?}"
  done
  st "prefill ok" ;;
*) st "unknown stage $STAGE"; exit 2 ;;
esac; done
