#!/usr/bin/env bash
# Append board rows (JSON lines on stdin) to bench/results/board.jsonl on main
# and push, from a throwaway worktree (the caller's checkout is untouched).
#   ... | scripts/board-push.sh "<what produced them>"
set -euo pipefail
rows=$(grep -v '^\s*$' || true)
[ -n "$rows" ] || exit 0
n=$(printf '%s\n' "$rows" | wc -l | tr -d ' ')
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
wt=$(mktemp -d)/board
git -C "$repo" fetch -q origin main
git -C "$repo" worktree add -q --detach "$wt" origin/main
trap 'git -C "$repo" worktree remove --force "$wt"' EXIT
for _ in 1 2 3; do
  printf '%s\n' "$rows" >> "$wt/bench/results/board.jsonl"
  git -C "$wt" commit -qs -m "board: $n row(s) from ${1:-a GPU run}" -- bench/results/board.jsonl
  if git -C "$wt" push -q origin HEAD:main; then
    echo "board: $n row(s) pushed to main" >&2
    exit 0
  fi
  # main moved: start again from it (only this throwaway worktree is reset)
  git -C "$wt" fetch -q origin main
  git -C "$wt" reset -q --hard origin/main
done
echo "board: push rejected 3 times" >&2
exit 1
