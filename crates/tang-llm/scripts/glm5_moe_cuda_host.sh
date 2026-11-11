#!/bin/sh
# Compile src/flash/glm5/moe_cuda.cuh as host C++ and run its kernels on the MoE fixtures.
# Usage (from the repo root): crates/tang-llm/scripts/glm5_moe_cuda_host.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/glm5_moe_cuda_host
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -o "$out" "$here/glm5_moe_cuda_host.cpp"
"$out" "$here/../tests/fixtures/glm5/moe"
