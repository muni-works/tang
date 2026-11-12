#!/usr/bin/env bash
# Run before merging a PR: merge origin/main into the current branch, push, then the
# post-merge guard's checks on the result (CPU build pod + coder-next smoke on H200 A).
#
#   scripts/premerge.sh            # current branch
#   SKIP_GPU=1 scripts/premerge.sh # CPU checks only (docs, scripts)
#
# Green: merge with `gh pr merge <n> --merge`. Red: fix on the branch first.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=guard/lib.sh
. "$here/guard/lib.sh"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" != main ] && [ "$branch" != HEAD ] || { echo "premerge: run on a PR branch" >&2; exit 2; }
git fetch -q origin main
if ! git merge -q --no-edit origin/main; then
  echo "premerge: merging origin/main conflicts; resolve, commit, rerun" >&2; exit 1
fi
git push -q --no-prune origin "HEAD:$branch"
sha=$(git rev-parse HEAD)
slot=pm-$(printf '%s' "$branch" | tr -c 'A-Za-z0-9' - | cut -c1-40)
echo "== premerge $branch @ ${sha:0:8}: CPU checks"
cpu_check "$sha" "$slot"
if [ "${SKIP_GPU:-0}" != 1 ]; then
  echo "== GPU smoke on $SMOKE_POD (coder-next)"
  gpu_smoke "$sha"
fi
echo "premerge: green at ${sha:0:8}"
