#!/usr/bin/env bash
# GLM-5.3-Flash all-GPU decode bench: one command, one summary line.
#
#   GLM5_FLAGS='TANG_GLM5_GEMV8=1 ...' scripts/glm5-bench.sh root@HOST -p PORT [source-dir]
#
# 1. rsync source-dir (default: this worktree) to /root/tang-bench/src; md5 the .rs +
#    Cargo.toml tree both sides and REFUSE on mismatch (stale-binary guard).
# 2. cargo build --release -p tang-llm --features cuda (skipped if GLM5_NO_BUILD=1).
# 3. weights: fetch unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL (6 shards, ~186 GB) if missing.
# 4. on the box, detached (scripts/glm5-bench-remote.sh), all-GPU (TANG_GLM5_GPU_FFN unset):
#    ids gate ('Write a haiku about the sea.' -n 8 --temp 0 must be the CPU oracle
#    785 1196 6801 264 6386 38038 911 279; abort on mismatch), then a timed decode
#    (-n GLM5_N=128 on a ~200-token prompt) with TANG_GLM5_TRACE=1.
# 5. prints: bench <commit> <flags>: ids OK, prefill X tok/s, decode Y tok/s, <stage> ms ...
#    (per decode token, wall). Logs: /root/tang-bench/out/<label>/{gate,bench}.log.
#
# GLM5_FLAGS  env assignments for both runs (empty = naive kernels). A/B = two invocations.
# GLM5_ROOT   remote work dir (default /root/tang-bench).
# GLM5_LABEL  output dir name (default: derived from the flags). GLM5_GGUF: first shard.
# Needs direct-TCP ssh (root@IP -p PORT); the ssh.runpod.io proxy is PTY-only.
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$PWD
DEST=${1:?usage: glm5-bench.sh root@HOST [-p PORT] [source-dir]}; shift
PORT=
if [ "${1:-}" = -p ]; then PORT=$2; shift 2; fi
SRC=$(cd "${1:-.}" && pwd)
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
SSH="ssh $SSHOPTS ${PORT:+-p $PORT} $DEST"
R=${GLM5_ROOT:-/root/tang-bench}
FLAGS=${GLM5_FLAGS:-}
LABEL=${GLM5_LABEL:-$(echo "${FLAGS:-naive}" | sed -E 's/TANG_GLM5_//g; s/[^A-Za-z0-9=]+/_/g')}
G5=${GLM5_GGUF:-/workspace/glm53/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
COMMIT=$(git -C "$SRC" rev-parse --short=8 HEAD 2>/dev/null || echo nogit)
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no -- crates 2>/dev/null)" ] || COMMIT="$COMMIT+dirty"

# 1. source up (+ this script's remote half, which may be newer than SRC's)
$SSH "mkdir -p $R/src"
rsync -a --delete --exclude target --exclude .git --exclude .frog --exclude .claude --exclude node_modules \
  ${PORT:+-e "ssh $SSHOPTS -p $PORT"} "$SRC/" "$DEST:$R/src/"
rsync -a ${PORT:+-e "ssh $SSHOPTS -p $PORT"} "$HERE/scripts/glm5-bench-remote.sh" "$DEST:$R/bench-remote.sh"
L=$(cd "$SRC" && find crates -name '*.rs' -o -name Cargo.toml | LC_ALL=C sort | xargs md5 -r | md5 -q)
RM=$($SSH "cd $R/src && find crates -name '*.rs' -o -name Cargo.toml | LC_ALL=C sort | xargs md5sum | awk '{print \$1\" \"\$2}' | md5sum | cut -c1-32")
echo "src md5: local=$L pod=$RM ($COMMIT)"
case "$RM" in *[!0-9a-f]*|'') echo "REFUSING: bad pod md5 output: $RM"; exit 1;; esac
[ "$L" = "$RM" ] || { echo "REFUSING: source md5 mismatch"; exit 1; }

# 2. build
if [ "${GLM5_NO_BUILD:-0}" != 1 ]; then
  $SSH "export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
[ -x /root/.cargo/bin/cargo ] || { curl -sSf https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal --default-toolchain stable >/dev/null; }
[ -x /usr/local/cuda/bin/nvcc ] || { echo 'REFUSING: no nvcc'; exit 1; }
cd $R/src && CUDA_PATH=/usr/local/cuda cargo build --release -p tang-llm --bin tang-llm --features cuda > $R/build.log 2>&1 \
  || { tail -30 $R/build.log; echo BUILD FAILED; exit 1; }
echo build ok: \$(md5sum target/release/tang-llm | cut -c1-32)"
fi

# 3. weights (all six shards)
$SSH "D=\$(dirname $G5); n=\$(ls \$D/*-of-00006.gguf 2>/dev/null | wc -l)
if [ \"\$n\" != 6 ]; then
  echo \"weights \$n/6 — fetching unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL (~30 min)\"
  command -v hf >/dev/null || pip install -q --break-system-packages -U huggingface_hub
  hf download unsloth/GLM-5.3-Flash-GGUF --include '*UD-Q4_K_XL*' --local-dir \$(dirname \$D) > $R/download.log 2>&1
  n=\$(ls \$D/*-of-00006.gguf | wc -l); [ \"\$n\" = 6 ] || { echo \"weights still \$n/6\"; exit 1; }
fi; echo weights 6/6"

# 4. run detached, poll
PROMPT=$(tr '\n' ' ' < "$HERE/scripts/glm5-bench-prompt.txt")
q() { printf '%q' "$1"; }
$SSH "rm -rf $R/out/$LABEL; mkdir -p $R/out; R=$R G5=$(q "$G5") FLAGS=$(q "$FLAGS") LABEL=$(q "$LABEL") COMMIT=$COMMIT N=${GLM5_N:-128} PROMPT=$(q "$PROMPT") \
  setsid nohup flock /root/gpu.lock bash $R/bench-remote.sh > $R/out/$LABEL.out 2>&1 < /dev/null & sleep 1; echo launched $LABEL"
until $SSH "test -f $R/out/$LABEL/DONE"; do sleep 30; done
$SSH "cat $R/out/$LABEL/summary"
