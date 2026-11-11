#!/bin/bash
# profile-when-ready.sh — waits for the GLM-5.3-Flash GGUF download to complete, then
# syncs nothing (the tree is already at ~/tang-scratch/glm5), builds CUDA, and runs ONE
# profiled window on the real model with TANG_GLM5_PROF=1. Result lands in
# ~/tang-scratch/profile-result.md. Runs unattended; safe to leave overnight.
set -u
LOG=~/tang-scratch/profile-when-ready.log
echo "$(date -u +%FT%TZ) armed" >> "$LOG"

# 1. wait for the download: the hf tool writes 'Downloaded' at completion; also require
#    all 6 shards present and the file count stable across two polls 60 s apart.
for i in $(seq 1 720); do
  if grep -q "Downloaded" ~/tang-scratch/dl.log 2>/dev/null; then
    N=$(ls ~/tang-scratch/glm53/UD-Q4_K_XL/*.gguf 2>/dev/null | wc -l)
    S1=$(du -sm ~/tang-scratch/glm53 2>/dev/null | cut -f1)
    sleep 60
    S2=$(du -sm ~/tang-scratch/glm53 2>/dev/null | cut -f1)
    if [ "$N" = "6" ] && [ "$S1" = "$S2" ]; then
      echo "$(date -u +%FT%TZ) download complete: $N shards, ${S2} MB" >> "$LOG"
      break
    fi
  fi
  [ "$i" = 720 ] && { echo "$(date -u +%FT%TZ) download never completed after 12h" >> "$LOG"; exit 1; }
  sleep 60
done

# 2. build CUDA (the tree at ~/tang-scratch/glm5 is already 7df4ba45 content + the
#    cuda-gated compile fixes merged at a156822c)
cd ~/tang-scratch/glm5
export PATH=/usr/local/cuda-12.8/bin:$HOME/.cargo/bin:$PATH
export CUDA_PATH=/usr/local/cuda-12.8
export CARGO_TARGET_DIR=$HOME/tang-scratch/target
cargo build --release -p tang-llm --no-default-features --features cuda >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "$(date -u +%FT%TZ) BUILD FAILED" >> "$LOG"
  tail -30 $CARGO_TARGET_DIR/../prof-build2.log >> "$LOG" 2>/dev/null
  exit 1
fi
echo "$(date -u +%FT%TZ) build ok" >> "$LOG"

# 3. ONE profiled window on the real model (shard 1 carries the metadata + first layers;
#    the mmap reads the rest). The 3090 holds attention cores + mixers; experts stay mmap'd.
BIN=$HOME/tang-scratch/target/release/tang-llm
G=$(find ~/tang-scratch/glm53/UD-Q4_K_XL -name '*-00001-of-*.gguf' | head -1)
echo "$(date -u +%FT%TZ) profiling $G" >> "$LOG"
TANG_GLM5_PROF=1 TANG_GPUS=2 TANG_GLM5_GPU_FFN=0 timeout 1800 $BIN flash-generate "$G" \
  --prompt "Write a haiku about the sea." -n 8 --temp 0 -v \
  > ~/tang-scratch/profile-run.log 2>&1
RC=$?
echo "$(date -u +%FT%TZ) flash-generate rc=$RC" >> "$LOG"

# 4. the report
{
  echo "# glm5 real-model profile — $(date -u +%FT%TZ)"
  echo
  echo '```'
  grep -E "prof |window t=1|prompt:|ids:|decode:|text:" ~/tang-scratch/profile-run.log
  echo '```'
} > ~/tang-scratch/profile-result.md
echo "$(date -u +%FT%TZ) result written to ~/tang-scratch/profile-result.md" >> "$LOG"
