# shellcheck shell=bash
# Shared by scripts/premerge.sh and scripts/main-guard.sh (sourced; runs on the Mac and
# drives the pods over ssh, so no credentials ever sit on a pod).
#   cpu_check <ref> <slot>   on the CPU build pod: fmt, CI's clippy, cuda build, tang-compile lib tests
#   gpu_smoke <ref>          on an H200 runner (auto pick, short lane): scripts/guard/smoke-coder.sh
# Both print a tail on failure and return nonzero. Override hosts with BUILD_SSH / SMOKE_POD.
BUILD_SSH=${BUILD_SSH:--p 38805 root@185.226.156.230}
SMOKE_POD=${SMOKE_POD:-auto}
GUARD_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cpu_check() {
  local ref=$1 slot=${2:-premerge}
  # shellcheck disable=SC2086
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 $BUILD_SSH \
    "REF='$ref' SLOT='$slot' bash -s" <<'POD'
set -uo pipefail
source /models/bootstrap/tang-up.sh >/dev/null 2>&1
rustup toolchain install 1.94.1 --profile minimal -c clippy -c rustfmt >/dev/null 2>&1
D=~/guard/$SLOT
[ -d "$D/.git" ] || git clone -q https://github.com/muni-works/tang "$D"
cd "$D" && git fetch -q origin "+refs/heads/*:refs/remotes/origin/*" \
  && git checkout -q -f "$REF" 2>/dev/null || git checkout -q -f "origin/$REF" || { echo "checkout $REF: FAIL"; exit 1; }
echo "commit: $(git log --oneline -1)"
export CARGO_TARGET_DIR=~/guard/target-$SLOT
step() { local name=$1; shift; if "$@" >/tmp/guard-$SLOT.log 2>&1; then echo "$name: ok"; else
  echo "$name: FAIL"; grep -E -A8 '^(error|test .* FAILED|failures:|thread .* panicked)' /tmp/guard-$SLOT.log | head -60
  [ -s /tmp/guard-$SLOT.log ] && tail -5 /tmp/guard-$SLOT.log; exit 1; fi; }
step fmt cargo +1.94.1 fmt --all -- --check
step clippy cargo +1.94.1 clippy --workspace --lib --tests --bins --exclude tang-site-wasm \
  --exclude tang-sheffer-wasm -- -D warnings -A clippy::too_many_arguments \
  -A clippy::type_complexity -A clippy::needless_range_loop -A clippy::ptr_arg
step "cuda build" cargo +1.94.1 build --release -p tang-bench --features cuda,gguf --examples
step "tang-compile tests" cargo +1.94.1 test -p tang-compile --lib
POD
}

gpu_smoke() {
  local ref=$1 job
  job=$(mktemp -t smoke).json
  printf '{"job":"cmd","model":"coder-next","cmd":"bash scripts/guard/smoke-coder.sh"}' >"$job"
  local res
  res=$(BOARD_PUSH=0 LANE=short "$GUARD_HERE/gpu-run.sh" "$SMOKE_POD" "$ref" "$job" 2>/dev/null)
  rm -f "$job"
  printf '%s' "$res" | python3 -c '
import json, sys
r = json.load(sys.stdin)
print((r.get("summary") or "").strip() or r.get("error") or "no summary")
sys.exit(0 if r.get("ok") else 1)'
}
