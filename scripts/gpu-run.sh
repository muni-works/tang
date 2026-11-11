#!/usr/bin/env bash
# One GPU job on a warm runner pod; blocks until it's done and prints result.json.
#
#   scripts/gpu-run.sh <pod> <ref> <job> [KEY=VAL ...]
#     pod   jx384ovrji8dj9 | q82kgiy7h4ifcu (or h200-1 / h200-2)
#     ref   branch, tag or commit (pushed to muni-works/tang)
#     job   check                  tiny parity + short Qwen3-8B timing (~minutes warm)
#           race:<model>           scripts/derived-race.sh (qwen3-8b) or
#                                  scripts/coder-derived-race.sh (coder-next)
#           race:<model>:<script>  any race script
#           cmd:<shell line>       anything in the checkout
#           file.json              a full job file (see scripts/gpu-runner/runner.py)
#     KEY=VAL  env for the job (LENS=128,4096 ...)
#   scripts/gpu-run.sh <pod> install    copy the runner to the pod and start it
#   scripts/gpu-run.sh <pod> status
# Pod addresses change on restart: override with GPU_RUN_SSH="-p PORT root@HOST".
set -euo pipefail
pod=${1:?pod}; shift
case $pod in
  jx384ovrji8dj9|h200-1) addr="-p 15450 root@157.66.255.42" ;;
  q82kgiy7h4ifcu|h200-2) addr="-p 19166 root@31.24.80.47" ;;
  *) addr=${GPU_RUN_SSH:?unknown pod $pod; set GPU_RUN_SSH} ;;
esac
addr=${GPU_RUN_SSH:-$addr}
# shellcheck disable=SC2086
s() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 $addr "$@"; }
here=$(cd "$(dirname "$0")" && pwd)

case ${1:?ref or install|status} in
  install)
    COPYFILE_DISABLE=1 tar --no-xattrs -C "$here/gpu-runner" -cf - . | s 'mkdir -p ~/runner/bin && tar -C ~/runner/bin -xf -'
    s "RESTART=${RESTART:-0} bash ~/runner/bin/start.sh"; exit ;;
  status) s 'python3 ~/runner/bin/runner.py status'; exit ;;
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
job=$(python3 -c 'import json,sys; j=json.loads(sys.argv[1]); j["ref"]=sys.argv[2]; j.setdefault("env",{}).update(json.loads(sys.argv[3])); print(json.dumps(j))' "$job" "$ref" "$envj")
id=$(printf '%s' "$job" | s 'python3 ~/runner/bin/runner.py submit -')
echo "job $id queued on $pod" >&2
# one server-side wait loop, 60 s between checks
s "d=~/runner/jobs/$id; until [ -f \$d/DONE ]; do sleep 60; done; cat \$d/result.json"
echo
echo "log: ssh $addr cat runner/jobs/$id/log.txt" >&2
