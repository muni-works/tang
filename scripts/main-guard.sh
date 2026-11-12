#!/usr/bin/env bash
# Post-merge guard: watches muni-works/tang main, and for each new tip runs the premerge
# checks (scripts/guard/lib.sh). On a failure it comments on the PR that commit merged,
# naming the commit and the failing step. Runs on the Mac (it holds the gh and ssh
# credentials; the pods hold none), e.g. in a terminal tab:
#
#   scripts/main-guard.sh          # loop, polls every ${GUARD_POLL:-120} s
#   scripts/main-guard.sh --once   # check the current tip once (no repeat of a seen tip)
#
# State: ~/.tang-main-guard/{last,log}. GitHub Actions is billing-blocked, hence this.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=guard/lib.sh
. "$here/guard/lib.sh"
REPO=${GUARD_REPO:-muni-works/tang}
ST=~/.tang-main-guard; mkdir -p "$ST"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$ST/log"; }

check() {
  local sha=$1 out rc=0 pr
  out=$({ echo "### CPU (tang-build)"; cpu_check "$sha" guard; } 2>&1) || rc=1
  if [ $rc = 0 ] && [ "${SKIP_GPU:-0}" != 1 ]; then
    out+=$'\n'$({ echo "### GPU smoke ($SMOKE_POD, coder-next)"; gpu_smoke "$sha"; } 2>&1) || rc=1
  fi
  printf '%s\n' "$out" >"$ST/${sha:0:12}.txt"
  if [ $rc = 0 ]; then log "${sha:0:8} green"; return 0; fi
  log "${sha:0:8} RED (details $ST/${sha:0:12}.txt)"
  pr=$(gh api "repos/$REPO/commits/$sha/pulls" -q '.[0].number' 2>/dev/null)
  local body
  body=$(printf '**main is red after this merge** (post-merge guard, `scripts/main-guard.sh`)\n\ncommit `%s`: the first failing step is below.\n\n```\n%s\n```\n\nReproduce from a branch with `scripts/premerge.sh`.' \
    "$sha" "$(printf '%s\n' "$out" | tail -80)")
  if [ -n "$pr" ]; then
    gh pr comment "$pr" -R "$REPO" -b "$body" >/dev/null && log "commented on #$pr"
  else
    gh api "repos/$REPO/commits/$sha/comments" -f body="$body" >/dev/null && log "commented on commit"
  fi
  return 1
}

while :; do
  sha=$(git ls-remote "https://github.com/$REPO" refs/heads/main | cut -f1)
  if [ -n "$sha" ] && [ "$sha" != "$(cat "$ST/last" 2>/dev/null)" ]; then
    log "main -> ${sha:0:8}"
    check "$sha"
    echo "$sha" >"$ST/last"
  fi
  [ "${1:-}" = --once ] && break
  sleep "${GUARD_POLL:-120}"
done
