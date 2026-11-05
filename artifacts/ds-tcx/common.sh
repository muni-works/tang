# sourced by the ds-tcx GPU scripts
set -euo pipefail
NAME="$(basename "$0" .sh)"
trap 'rc=$?; if [ $rc -eq 0 ]; then echo "DONE $NAME"; else echo "FAIL $NAME (exit $rc, line $LINENO)"; fi' EXIT
ROOT="${ROOT:-$HOME/build/ds-tcx}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/target-ds-tcx}"
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"
export PATH=/usr/local/cuda/bin:$PATH
export DS41_TINY="${DS41_TINY:-/tmp/ds41-tiny}"
CKPT="${CKPT:-/workspace/deepseek-v41-flash}"
LOG="${LOG:-$ROOT/artifacts/ds-tcx/logs}"; mkdir -p "$LOG"
cd "$ROOT"
