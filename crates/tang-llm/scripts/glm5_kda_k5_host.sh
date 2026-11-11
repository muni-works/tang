#!/bin/sh
# Compile src/flash/glm5/kda_k5.cuh (+ the MoE source for G5W/g5_deq/g5_dot) as host C++ and
# run the KDA projection kernels on the kda fixtures.
# Usage (from the repo root): crates/tang-llm/scripts/glm5_kda_k5_host.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/glm5_kda_k5_host
${CXX:-c++} -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -o "$out" "$here/glm5_kda_k5_host.cpp"
"$out" "$here/../tests/fixtures/glm5/kda"
