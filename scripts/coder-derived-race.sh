#!/usr/bin/env bash
# Qwen3-Coder-Next at real size: the derived decode step (claude/derive-integrate,
# `examples/coder_derived.rs`) against the hand coder engine (`tang-llm
# flash-generate` / `flash-bench`) and the CPU f32 reference (`tang-llm flash-ref`),
# on unsloth's UD-Q4_K_XL GGUF.
#
#   1. build tang-llm (cuda) and coder_derived (cuda, gguf)
#   2. weights: Qwen3-Coder-Next-UD-Q4_K_XL.gguf (49.6 GB) onto container disk
#   3. greedy, 64 tokens after a fixed 21-token prompt: hand engine ids; derived
#      ids, its per-position top-5 logprobs, and device ms per step at $LENS
#   4. flash-ref over prompt + the derived tokens (teacher-forced, top-5 at the
#      last 64 positions): top-1 agreement and the top-1 logprob gap
#   5. hand decode tok/s (flash-bench) next to the derived device step time
#
# Run from the repo root:  bash scripts/coder-derived-race.sh
# Needs: one CUDA GPU with >= 80 GB (H100 80GB, H200, RTX PRO 6000 96GB, B200):
# the derived step holds ~52 GB of weights (49.6 GB as stored, token_embd and
# blk.46's Q6_K/BF16 tensors as f32), the hand engine ~50 GB; they run one after
# the other. ~120 GB container disk (model 50 GB, cargo target ~15 GB), >= 64 GB
# host RAM (the GGUF is mapped; flash-ref dequantizes per tensor). Time: about
# 1.5 h: download 5-15 min, build 15-20 min, derived step build + upload a few
# minutes, hand generate a few minutes, flash-ref on the CPU the longest
# (estimate 20-40 min for 85 positions of an 80B-A3B model).
#
# Env: MODEL_DIR [/root/models/coder-next], OUT [/root/coder-derived-race],
# LENS [128,4096], CTX [4352], SKIP_REF=1 skips flash-ref.
set -euo pipefail

REPO=${REPO:-$PWD}
MODEL_DIR=${MODEL_DIR:-/root/models/coder-next}
OUT=${OUT:-/root/coder-derived-race}
LENS=${LENS:-128,4096}
CTX=${CTX:-4352}
GGUF="$MODEL_DIR/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
mkdir -p "$OUT" "$MODEL_DIR"
cd "$REPO"

log() { printf '\n=== %s\n' "$*"; }

log "toolchain and GPU"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"
{ git log --oneline -1 2>/dev/null || echo "no git checkout"; } | tee "$OUT/commit.txt"

log "weights: unsloth/Qwen3-Coder-Next-GGUF UD-Q4_K_XL -> $GGUF"
if [ ! -f "$GGUF" ]; then
  # the RunPod PyTorch image's system Python is "externally managed"
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q "huggingface_hub[hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  if command -v hf >/dev/null; then
    hf download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-Q4_K_XL.gguf --local-dir "$MODEL_DIR"
  else
    huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-Q4_K_XL.gguf --local-dir "$MODEL_DIR"
  fi
fi
test -f "$GGUF"

log "build"
time cargo build --release -p tang-llm --features cuda --bin tang-llm 2>&1 | tail -2
time cargo build --release -p tang-bench --features cuda,gguf --example coder_derived 2>&1 | tail -2
LLM=target/release/tang-llm
DER=target/release/examples/coder_derived

# a fixed 21-token prompt (ids < vocab; any ids will do, all runs share them)
echo "785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 13 576 6722 315 15344 374 21718 13" \
  >"$OUT/prompt.ids"

log "hand coder engine: greedy 64 and decode speed after the prompt (flash-bench)"
# (flash-generate runs the Flash-Next engine; flash-bench picks the qwen3next Coder)
"$LLM" flash-bench "$GGUF" --ids-file "$OUT/prompt.ids" -n 64 --draft none \
  2>&1 | tee "$OUT/hand-bench.txt" || echo "hand bench failed"
grep -E '^ids:' "$OUT/hand-bench.txt" | sed 's/^ids: //' >"$OUT/ids-hand.txt" || true

log "derived step: greedy 64, logprobs, device step time"
"$DER" "$GGUF" --ids-file "$OUT/prompt.ids" -n 64 --ctx "$CTX" --top 5 \
  --logprobs "$OUT/derived.jsonl" --out "$OUT/ids-derived.txt" --bench "$LENS" \
  2>&1 | tee "$OUT/derived.txt" || echo "derived run failed"

log "hand decode speed at 4096 context"
"$LLM" flash-bench "$GGUF" --ids-file "$OUT/prompt.ids" --ctx 4096 -n 64 --draft none \
  2>&1 | tee "$OUT/hand-bench-4096.txt" || echo "hand bench 4096 failed"

if [ "${SKIP_REF:-0}" != 1 ] && [ -s "$OUT/ids-derived.txt" ]; then
  log "flash-ref (CPU f32) over prompt + derived tokens"
  # positions: last prompt token, then every generated token but the last
  cat "$OUT/prompt.ids" "$OUT/ids-derived.txt" | tr '\n' ' ' | awk '{NF--; print}' >"$OUT/ref.ids"
  time "$LLM" flash-ref "$GGUF" --ids-file "$OUT/ref.ids" --top 5 --last 64 \
    >"$OUT/ref.jsonl" 2>"$OUT/ref.err" || echo "flash-ref failed (see $OUT/ref.err)"
fi

log "summary"
{
  echo "GPU: $(tail -1 "$OUT/gpu.txt")"
  echo "commit: $(cat "$OUT/commit.txt")"
  echo
  if [ -s "$OUT/ids-hand.txt" ] && [ -s "$OUT/ids-derived.txt" ]; then
    python3 - "$OUT/ids-hand.txt" "$OUT/ids-derived.txt" <<'EOF'
import sys
h = open(sys.argv[1]).read().split()
d = open(sys.argv[2]).read().split()
same = 0
for a, b in zip(h, d):
    if a != b:
        break
    same += 1
print(f"greedy ids, derived vs hand: first {same} of {min(len(h), len(d))} identical")
EOF
  fi
  if [ -s "$OUT/ref.jsonl" ]; then
    python3 - "$OUT/ref.jsonl" "$OUT/derived.jsonl" <<'EOF'
import json, sys
ref = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
der = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
n = min(len(ref), len(der))
ref, der = ref[-n:], der[:n]
agree = sum(r["top"][0][0] == d["top"][0][0] for r, d in zip(ref, der))
gap = max(abs(r["top"][0][1] - dict((i, v) for i, v in d["top"]).get(r["top"][0][0], -99))
          for r, d in zip(ref, der))
print(f"vs flash-ref (f32 CPU): top-1 agrees at {agree}/{n} positions; "
      f"max |logprob gap| of the reference's top-1 {gap:.4f}")
EOF
  fi
  grep -E '^derived decode step at len' "$OUT/derived.txt" | sed 's/^/  /' || true
  grep -E '^  decode ' "$OUT/hand-bench.txt" | sed 's/^/  hand (ctx 21):/' || true
  grep -E '^  decode ' "$OUT/hand-bench-4096.txt" | sed 's/^/  hand (ctx 4096):/' || true
  grep -E 'end to end' "$OUT/derived.txt" | sed 's/^/  derived: /' || true
} | tee "$OUT/summary.txt"
