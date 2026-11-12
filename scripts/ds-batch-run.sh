#!/bin/bash
# The DeepSeek V4.1 Flash (1017) batching run on an 8-GPU box (8x H100 80 GB / H200 / B200):
# download the weights to local disk, build tang with expert parallelism, run the
# single-stream baseline and the --slots sweep, and print aggregate tok/s and $/M output tokens.
#
# Run it detached as soon as the box is up (nothing here waits on a terminal):
#   scp -P <port> ds-batch-run.sh tang-src.tgz root@<host>:/root/
#   ssh -p <port> root@<host> 'nohup bash /root/ds-batch-run.sh > /root/ds-run.log 2>&1 &'
# then one server-side wait loop on /root/ds-run/DONE (poll every 60 s or slower).
#
# Inputs (environment):
#   PRICE_HR    the box's $/hr, for the $/M column (default 31.92 = 8x H100 SECURE)
#   SRC_TGZ     tang source tarball (default /root/tang-src.tgz, made with `git archive`); if
#               absent, GH_TOKEN + BRANCH clone muni-works/tang over HTTPS instead
#   BRANCH      default claude/ds-multiseq
#   HF_TOKEN    if the weights repo needs one
#   SLOTS       the sweep (default "8 16 32 64"), CHUNKS the DS_ROWS_CHUNK values (default "8 0")
#   STEPS / PROMPT  decode steps per slot (200) and prompt tokens per slot (512)
#
# Disk: 510 GB of weights (48 shards, 96,085 tensors) + ~15 GB toolchain/build: give the
# container disk >= 600 GB and keep the weights OFF a network volume (RunPod volumes cap near
# 13 GB/min; container disk with hf_transfer has run at ~1 GB/s: expect 10-20 min, about 1 h on
# a volume). Host RAM: the experts are pinned once per process (~200 GB).
set -u
OUT=/root/ds-run
W=/root/ds1017
PRICE_HR=${PRICE_HR:-31.92}
BRANCH=${BRANCH:-claude/ds-multiseq}
SLOTS=${SLOTS:-"8 16 32 64"}
CHUNKS=${CHUNKS:-"8 0"}
STEPS=${STEPS:-200}
PROMPT=${PROMPT:-512}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/progress.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$OUT/DONE"; exit 1; }

# ---- preflight
NG=$(nvidia-smi -L | wc -l)
[ "$NG" -ge 8 ] || fail "need 8 GPUs, found $NG"
FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 560 ] || fail "need >= 560 GB free on /, have ${FREE} GB"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpus.csv"
log "preflight ok: $NG GPUs, ${FREE} GB free"

# ---- weights download (background) and build (foreground) in parallel
(
  pip install -q --break-system-packages "huggingface_hub[hf_transfer]" hf_transfer 2>/dev/null \
    || pip install -q "huggingface_hub[hf_transfer]" hf_transfer
  export HF_HUB_ENABLE_HF_TRANSFER=1
  t0=$(date +%s)
  hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$W" > "$OUT/download.log" 2>&1
  rc=$?
  echo "rc=$rc secs=$(( $(date +%s) - t0 )) bytes=$(du -sb "$W" | cut -f1)" > "$OUT/download.done"
) &
DL=$!

log "build: rustup + tang ($BRANCH)"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/rustup.log" 2>&1 || fail rustup
fi
source /root/.cargo/env
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq pkg-config libssl-dev clang >/dev/null 2>&1
mkdir -p /root/w && cd /root/w
if [ -f "${SRC_TGZ:-/root/tang-src.tgz}" ]; then
  rm -rf tang && mkdir tang && tar xzf "${SRC_TGZ:-/root/tang-src.tgz}" -C tang
else
  [ -n "${GH_TOKEN:-}" ] || fail "no source tarball and no GH_TOKEN"
  git clone -q --branch "$BRANCH" "https://x-access-token:${GH_TOKEN}@github.com/muni-works/tang.git" tang || fail clone
fi
cd /root/w/tang
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
# libnccl.so.2 for --features ep: the pytorch image ships it with the nvidia-nccl wheel
NCCL=$(python3 -c 'import nvidia.nccl,os;print(os.path.dirname(nvidia.nccl.__file__)+"/lib")' 2>/dev/null)
[ -n "$NCCL" ] && export LD_LIBRARY_PATH="$NCCL:${LD_LIBRARY_PATH:-}"
ls "$NCCL"/libnccl.so* >/dev/null 2>&1 || ldconfig -p | grep -q libnccl.so.2 || fail "no libnccl.so.2"
cargo build -p tang-llm --release --no-default-features --features ep --bins > "$OUT/build.log" 2>&1 \
  || fail "build (see build.log)"
BIN=/root/w/tang/target/release/tang-llm
log "build ok: $(md5sum $BIN | cut -c1-8)"

log "waiting for the weights download"
wait $DL
cat "$OUT/download.done" | tee -a "$OUT/progress.log"
grep -q '^rc=0' "$OUT/download.done" || fail "download (see download.log)"
N=$(ls "$W"/*.safetensors 2>/dev/null | wc -l)
[ "$N" -ge 48 ] || fail "expected 48 shards, have $N"

DEV=0,1,2,3,4,5,6,7
cd "$OUT"
run() { # name, then the command; one log per run, never two at once
  local name=$1; shift
  log "run $name"
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  log "  $name rc=$? in $(( $(date +%s) - t0 )) s: $(grep -E 'aggregate|tok/s' "$OUT/$name.log" | tail -1)"
}
# baseline: single stream, greedy, no drafts (like for like with the batched decode)
run single "$BIN" deepseek-profile "$W" --devices $DEV --no-spec --steps "$STEPS" --prompt-tokens "$PROMPT"
# the shipped single stream (DSpark drafts), for reference
run single-spec "$BIN" deepseek-profile "$W" --devices $DEV --steps 40 --prompt-tokens "$PROMPT"
for c in $CHUNKS; do
  for n in $SLOTS; do
    DS_ROWS_CHUNK=$c DS_SLOT_CTX=$(( PROMPT + STEPS + 64 )) \
      run "slots$n-chunk$c" "$BIN" deepseek-profile "$W" --devices $DEV --slots "$n" \
      --steps "$STEPS" --prompt-tokens "$PROMPT"
  done
done

# ---- summary: aggregate tok/s -> $/M output tokens at PRICE_HR
{
  echo "run,aggregate_tok_s,usd_per_M_output"
  for f in "$OUT"/slots*.log; do
    a=$(grep -oE 'aggregate [0-9.]+' "$f" | awk '{print $2}')
    [ -n "$a" ] && echo "$(basename "$f" .log),$a,$(awk -v a="$a" -v p="$PRICE_HR" 'BEGIN{printf "%.3f", p/(3600*a)*1e6}')"
  done
} > "$OUT/summary.csv"
log "summary:"; cat "$OUT/summary.csv" | tee -a "$OUT/progress.log"
# oracle check: slot 0's ids under batching must equal the single stream's first ids
grep -h '^ids slot 0' "$OUT"/slots*.log | sort | uniq -c > "$OUT/slot0-ids.txt"
log "slot 0 ids across runs (expect one distinct line):"; cat "$OUT/slot0-ids.txt" | tee -a "$OUT/progress.log"
echo ok > "$OUT/DONE"
