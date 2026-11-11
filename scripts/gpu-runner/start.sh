#!/usr/bin/env bash
# Install (idempotent) and start the runner on this pod; a running runner is left alone
# unless RESTART=1. Run from the directory holding runner.py (gpu-run.sh copies it to
# ~/runner/bin and runs this there).
set -euo pipefail
HOME_R=${RUNNER_HOME:-$HOME/runner}
BIN=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HOME_R/jobs"
# toolchain + sccache from the volume (muni-works/tang#226); clones tang into src/
TANG_DIR=$HOME_R/src bash /models/bootstrap/tang-up.sh >/dev/null
pid=$(cat "$HOME_R/runner.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  [ "${RESTART:-0}" = 1 ] || { echo "runner already running (pid $pid)"; exit 0; }
  kill "$pid"; sleep 1
fi
cd "$HOME_R"
setsid nohup python3 "$BIN/runner.py" serve >>"$HOME_R/runner.log" 2>&1 </dev/null &
sleep 1
echo "runner started (pid $(cat "$HOME_R/runner.pid"))"
