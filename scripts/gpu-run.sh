#!/usr/bin/env bash
# One GPU job on a warm runner pod; blocks until it's done and prints result.json.
#
#   scripts/gpu-run.sh <pod> <ref> <job> [KEY=VAL ...]
#     pod   w2myvdc7bi8mke | ez4ue9z5y6yai3 (or h200-1 = prefill/baselines,
#           h200-2 = coder serving/decode; auto = whichever H200 starts the job first)
#     ref   branch, tag or commit (pushed to muni-works/tang)
#     job   check                  tiny parity + short Qwen3-8B timing (~minutes warm)
#           race:<model>           scripts/derived-race.sh (qwen3-8b) or
#                                  scripts/coder-derived-race.sh (coder-next)
#           race:<model>:<script>  any race script
#           cmd:<shell line>       anything in the checkout
#           file.json              a full job file (see scripts/gpu-runner/runner.py)
#     KEY=VAL  env for the job (LENS=128,4096 ...)
#   LANE=short   jump the queue past long jobs (smokes, quick checks: minutes, not an
#                hour); a check is short already. A running job is never interrupted.
#   scripts/gpu-run.sh <pod> install    copy the runner to the pod and start it
#   scripts/gpu-run.sh <pod> status
# Pod addresses change on restart: override with GPU_RUN_SSH="-p PORT root@HOST".
set -euo pipefail
pod=${1:?pod}; shift
H200S="w2myvdc7bi8mke ez4ue9z5y6yai3"
# auto: the H200 where a job of this lane starts soonest (estimates: a short job 5 min,
# a long one 30, a running long one 15 more); ties go to h200-2 (A runs the long prefills)
if [ "$pod" = auto ]; then
  lane=${LANE:-}; case ${2:-} in check) lane=${lane:-short} ;; esac; lane=${lane:-long}
  best=; best_w=
  for p in $H200S; do
    st=$(GPU_RUN_SSH='' "$0" "$p" status --json 2>/dev/null | tail -1) || continue
    w=$(printf '%s' "$st" | python3 -c '
import json, sys
s, lane = json.load(sys.stdin), sys.argv[1]
est = {"short": 5, "long": 30}
r = s.get("running")
w = 0 if not r else (est["short"] if r["lane"] == "short" else 15)
w += sum(est[j["lane"]] for j in s["pending"] if lane == "long" or j["lane"] == "short")
print(w)' "$lane") || continue
    if [ -z "$best_w" ] || [ "$w" -le "$best_w" ]; then best=$p best_w=$w; fi
  done
  pod=${best:?auto: no H200 runner answered}
  echo "auto: $pod (about ${best_w} min of $lane-lane work ahead)" >&2
fi
case $pod in
  w2myvdc7bi8mke|h200-1) addr="-p 11865 root@213.181.111.134" ;;
  ez4ue9z5y6yai3|h200-2) addr="-p 13490 root@103.196.86.36" ;;
  *) addr=${GPU_RUN_SSH:?unknown pod $pod; set GPU_RUN_SSH} ;;
esac
addr=${GPU_RUN_SSH:-$addr}
# shellcheck disable=SC2086
s() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 $addr "$@"; }
here=$(cd "$(dirname "$0")" && pwd)

case ${1:?ref or install|status} in
  install)
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$here/gpu-runner" -cf - . -C "$here" bench_board.py | s 'mkdir -p ~/runner/bin && tar -C ~/runner/bin -xf -'
    s "RESTART=${RESTART:-0} bash ~/runner/bin/start.sh"; exit ;;
  status) s "python3 ~/runner/bin/runner.py status ${2:-}"; exit ;;
esac
ref=$1 spec=${2:?job}; shift 2
envj=$(for kv in "$@"; do printf '%s\n' "$kv"; done | python3 -c '
import json,sys; print(json.dumps(dict(l.rstrip("\n").split("=",1) for l in sys.stdin if "=" in l)))')
case $spec in
  check) job="{\"job\":\"check\"}" ;;
  race:*:*) m=${spec#race:}; job=$(python3 -c 'import json,sys; print(json.dumps({"job":"race","model":sys.argv[1],"script":sys.argv[2]}))' "${m%%:*}" "${m#*:}") ;;
  race:coder-next) job='{"job":"race","model":"coder-next","script":"scripts/coder-derived-race.sh"}' ;;
  race:*) job="{\"job\":\"race\",\"model\":\"${spec#race:}\",\"script\":\"scripts/derived-race.sh\"}" ;;
  cmd:*) job=$(python3 -c 'import json,sys; print(json.dumps({"job":"cmd","cmd":sys.argv[1]}))' "${spec#cmd:}") ;;
  *.json) job=$(cat "$spec") ;;
  *) echo "job? $spec" >&2; exit 2 ;;
esac
job=$(python3 -c 'import json,sys; j=json.loads(sys.argv[1]); j["ref"]=sys.argv[2]; j.setdefault("env",{}).update(json.loads(sys.argv[3]))
if sys.argv[4]: j["lane"]=sys.argv[4]
print(json.dumps(j))' "$job" "$ref" "$envj" "${LANE:-}")
id=$(printf '%s' "$job" | s 'python3 ~/runner/bin/runner.py submit -')
echo "job $id queued on $pod" >&2
# one server-side wait loop, 60 s between checks
result=$(s "d=~/runner/jobs/$id; until [ -f \$d/DONE ]; do sleep 60; done; cat \$d/result.json")
printf '%s\n' "$result"
echo "log: ssh $addr cat runner/jobs/$id/log.txt" >&2
# the job's board rows go to bench/results/board.jsonl on main (BOARD_PUSH=0: don't)
if [ "${BOARD_PUSH:-1}" != 0 ]; then
  printf '%s' "$result" | python3 -c 'import json,sys
for r in json.load(sys.stdin).get("board") or []: print(json.dumps(r))' \
    | "$here/board-push.sh" "gpu-runner $pod job $id ($ref)" || echo "board: push failed" >&2
fi
