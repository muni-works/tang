#!/usr/bin/env bash
# Compile one of tang-compute's CUDA kernel source constants offline and report what ptxas
# made of it: registers, spills, and counts of the instructions a tensor-core kernel should
# (HMMA, LDSM) and should not (LDL/STL: local-memory spills) contain.
#
#   scripts/sass_check.sh F16_GEMM_CUDA [sm_86]
#
# Needs nvcc on PATH (run on the CUDA host). Fails if the kernel spills or has no HMMA.
set -euo pipefail
name=${1:?usage: sass_check.sh CONST_NAME [arch]}
arch=${2:-sm_86}
src=crates/tang-compute/src/kernels/llm_cuda.rs
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 -I - "$src" "$name" > "$tmp/k.cu" <<'EOF'
import sys
s = open(sys.argv[1]).read()
name = sys.argv[2]
common = s.split('concat!(\n            r#"', 1)[1].split('"#', 1)[0]
head = f'pub const {name}: &str = '
if head not in s:
    sys.exit(f'{name} not found')
rest = s.split(head, 1)[1]
wrapped = rest.startswith('with_common!(')
body = rest.split('r#"', 1)[1].split('"#', 1)[0]
print((common if wrapped else '') + body)
EOF
nvcc -std=c++17 -cubin -arch="$arch" -Xptxas -v "$tmp/k.cu" -o "$tmp/k.cubin" 2>&1 |
    grep -E 'Function properties|registers|spill' || true
cuobjdump -sass "$tmp/k.cubin" > "$tmp/k.sass"
for op in HMMA IMMA LDSM LDGSTS LDG STS LDL STL BAR; do
    printf '%-8s %s\n' "$op" "$(grep -c "$op" "$tmp/k.sass" || true)"
done
spills=$(grep -cE '\b(LDL|STL)\b' "$tmp/k.sass" || true)
mma=$(grep -cE 'HMMA|IMMA' "$tmp/k.sass" || true)
[ "$spills" -eq 0 ] || { echo "FAIL: $spills local-memory ops (spills)"; exit 1; }
[ "$mma" -gt 0 ] || { echo "FAIL: no tensor-core instructions"; exit 1; }
echo ok
