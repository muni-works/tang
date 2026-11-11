#!/usr/bin/env bash
# GLM-5.3-Flash all-GPU bench: one command for a GPU box (the B300 trip).
#
#   scripts/glm5-b300-bench.sh user@HOST [-p PORT] [--no-wait | --status]
#
# 1. ships the COMMITTED tree (HEAD): git archive -> scp (never a stdin pipe into ssh,
#    it stalls on the pod links). Refuses a dirty worktree, so `git ls-files` = HEAD.
# 2. md5s every `git ls-files` file locally; the box re-checks each one in its source
#    dir (md5sum -c) and refuses on any mismatch, missing or extra file.
# 3. starts scripts/glm5-b300-bench-remote.sh FROM THE VERIFIED TREE, detached (setsid
#    nohup), so a dropped link doesn't kill it. It does the cuda build (binary md5 before
#    / after; aborts on a failed or unchanged build), fetches the weights if missing (in
#    parallel with the build), then under `flock ~/.cache/tang-gpu.lock`, all-GPU
#    (TANG_GLM5_MOE=fast, every fast flag, no TANG_GLM5_GPU_FFN=0):
#      - single stream, -n 128, TANG_GLM5_TRACE=1: the ids gate (first 8 = the oracle)
#        and decode tok/s;
#      - --slots 8 / 32 --diverse: aggregate tok/s (slot 0's ids = the oracle prefix).
#    Writes <remote>/summary.md and <remote>/status (last line DONE, also on abort).
# 4. waits (polls status every 60 s over short ssh calls) and prints the summary.
#    --no-wait: launch and exit. --status: print status + summary of the last run.
#
# Needs a direct-TCP ssh (RunPod "SSH over exposed TCP": root@IP -p PORT). The
# ssh.runpod.io proxy is PTY-only and rejects commands and scp.
#
# Env (all optional):
#   GLM5_GGUF      first shard (or any glm5 gguf). Default
#                  /workspace/glm53/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf;
#                  the default path is fetched (unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL,
#                  6 shards, ~186 GB, ~30 min on the B300) if any shard is missing.
#   GLM5_ORACLE    expected greedy ids (default the CPU oracle 785 1196 6801 264 6386
#                  38038 911 279), or `cpu`: compute it with a CPU run (TANG_GLM5_GPU=0,
#                  -n 8) first — for small fixtures only.
#   GLM5_PROMPT    default 'Write a haiku about the sea.'; GLM5_PROMPT_IDS overrides it.
#   GLM5_N_SINGLE  single-stream tokens (128). GLM5_N_SLOTS: tokens per slot (16).
#   GLM5_SLOTS     slot counts ("8 32").
#   GLM5_ENV       extra env for every run, e.g. 'TANG_GLM5_KDA_MAXT=64'.
#   GLM5_REMOTE    remote work dir (default $HOME/tang-bench on the box).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?usage: glm5-b300-bench.sh user@HOST [-p PORT] [--no-wait | --status]}; shift
PORT= MODE=wait
while [ $# -gt 0 ]; do
  case "$1" in
    -p) PORT=$2; shift 2 ;;
    --no-wait) MODE=nowait; shift ;;
    --status) MODE=status; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
SSHOPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
  -o ServerAliveInterval=30 -o ServerAliveCountMax=20)
ssh_() { ssh "${SSHOPTS[@]}" ${PORT:+-p "$PORT"} "$DEST" "$@" </dev/null; }
# a small bash script as a base64 ARGUMENT (no stdin), run by bash whatever the login shell
rsh() { ssh_ "echo $(printf '%s' "$1" | base64 | tr -d '\n') | base64 -d | bash"; }

RHOME=$(ssh_ 'echo $HOME') || { echo "REFUSING: ssh $DEST ${PORT:+-p $PORT} failed (need direct-TCP ssh, not the ssh.runpod.io proxy)"; exit 1; }
case "$RHOME" in /*) ;; *) echo "REFUSING: bad remote \$HOME: $RHOME"; exit 1 ;; esac
R=${GLM5_REMOTE:-$RHOME/tang-bench}

show() {
  rsh "cat '$R/status' 2>/dev/null || echo '(no status yet)'; echo; [ -f '$R/summary.md' ] && cat '$R/summary.md' || true"
}
if [ "$MODE" = status ]; then show; exit 0; fi

# 1. the committed tree
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "REFUSING: tracked files modified — commit first (this ships HEAD only):"; git status --short --untracked-files=no; exit 1; }
TREE=$(git rev-parse HEAD)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git archive --format=tar.gz -o "$T/src.tgz" HEAD
# "md5  path" (md5sum -c format) for every tracked file
git ls-files -z | xargs -0 md5 -r | sed 's/ /  /' | LC_ALL=C sort -k2 > "$T/manifest"
NF=$(wc -l < "$T/manifest" | tr -d ' ')
TGZ=$(md5 -q "$T/src.tgz")
q() { printf '%q' "$1"; }
{
  echo "TREE=$(q "$TREE")"
  echo "NF=$(q "$NF")"
  echo "TGZ_MD5=$(q "$TGZ")"
  echo "R=$(q "$R")"
  echo "G5=$(q "${GLM5_GGUF:-/workspace/glm53/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}")"
  echo "ORACLE=$(q "${GLM5_ORACLE:-785 1196 6801 264 6386 38038 911 279}")"
  echo "PROMPT=$(q "${GLM5_PROMPT:-Write a haiku about the sea.}")"
  echo "PROMPT_IDS=$(q "${GLM5_PROMPT_IDS:-}")"
  echo "N_SINGLE=$(q "${GLM5_N_SINGLE:-128}")"
  echo "N_SLOTS=$(q "${GLM5_N_SLOTS:-16}")"
  echo "SLOTS=$(q "${GLM5_SLOTS:-8 32}")"
  echo "XENV=$(q "${GLM5_ENV:-}")"
} > "$T/bench.env"
echo "tree $TREE ($NF files, src.tgz $TGZ)"

rsh "mkdir -p '$R/in' && rm -f '$R/in/src.tgz' '$R/in/manifest' '$R/in/bench.env'"
scp -q "${SSHOPTS[@]}" ${PORT:+-P "$PORT"} "$T/src.tgz" "$T/manifest" "$T/bench.env" "$DEST:$R/in/" </dev/null

# 2. unpack into a stage dir, copy only CHANGED files into src (unchanged files keep
#    their mtimes, so cargo rebuilds what changed), drop extras, then md5 every file.
#    target/ lives outside src, so src is exactly the tree.
SYNC=$(cat <<'EOF'
set -euo pipefail
cd "$R"
[ "$(md5sum in/src.tgz | cut -c1-32)" = "$TGZ" ] || { echo "REFUSING: src.tgz md5 mismatch"; exit 1; }
rm -rf stage && mkdir -p stage src && tar -xzf in/src.tgz -C stage
cut -c35- in/manifest > in/paths
kind() { if [ -L "$1" ]; then echo l; elif [ -e "$1" ]; then echo f; else echo -; fi; }
n=0
while IFS= read -r p; do
  if [ "$(kind "stage/$p")" != "$(kind "src/$p")" ] || ! cmp -s "stage/$p" "src/$p"; then
    mkdir -p "src/$(dirname "$p")"; rm -f "src/$p"; cp -P "stage/$p" "src/$p"; n=$((n + 1))
  fi
done < in/paths
(cd src && find . \( -type f -o -type l \) | sed 's|^\./||' | LC_ALL=C sort) > in/have
LC_ALL=C sort in/paths > in/want
LC_ALL=C comm -23 in/have in/want > in/extra
(cd src && while IFS= read -r p; do rm -f "$p"; done < ../in/extra)
(cd src && md5sum -c --quiet ../in/manifest) || { echo "REFUSING: md5 mismatch in src"; exit 1; }
have=$(cd src && find . \( -type f -o -type l \) | wc -l)
[ "$have" = "$NF" ] || { echo "REFUSING: src has $have files, manifest $NF"; exit 1; }
rm -rf stage
echo "src md5-verified on the box: $NF/$NF files ($n updated, $(wc -l < in/extra) removed), tree $TREE"
EOF
)
rsh "R='$R' TGZ='$TGZ' NF='$NF' TREE='$TREE'
$SYNC"

# 3. launch the remote half from the verified tree, detached
rsh "cd '$R' && rm -f status summary.md && setsid nohup bash src/scripts/glm5-b300-bench-remote.sh in/bench.env > run.out 2>&1 < /dev/null & sleep 2; cat '$R/status' 2>/dev/null; cat '$R/run.out'"
echo "launched: status $DEST:$R/status, summary $DEST:$R/summary.md"
[ "$MODE" = nowait ] && exit 0

# 4. wait for DONE
seen=0
while :; do
  s=$(rsh "cat '$R/status' 2>/dev/null" || true)
  n=$(printf '%s\n' "$s" | grep -c . || true)
  [ "$n" -gt "$seen" ] && { printf '%s\n' "$s" | tail -n +"$((seen + 1))"; seen=$n; }
  printf '%s\n' "$s" | tail -1 | grep -qx DONE && break
  sleep 60
done
echo; rsh "cat '$R/summary.md'"
