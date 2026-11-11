#!/bin/bash
# usage: run.sh "UNROLL:MINB:LDU ..." [gun-only]   (on the GPU box, next to roofline.cu + moe_cuda.cuh)
set -e
cd "$(dirname "$0")"
for cfg in ${1:-1:8:1}; do
  IFS=: read un mb ldu <<< "$cfg"
  sed -e "s/#pragma unroll G5F_UNROLL/#pragma unroll $un/" -e "s/#pragma unroll G5F_LDU/#pragma unroll $ldu/" moe_cuda.cuh > moe_cuda_b.cuh
  sed "s/moe_cuda.cuh/moe_cuda_b.cuh/" roofline.cu > roofline_b.cu
  /usr/local/cuda/bin/nvcc -O3 -arch=sm_100 -std=c++17 -DG5F_MINB=$mb -Xptxas -v -I. roofline_b.cu -o roofline_$cfg 2>&1 | grep -A1 "g5f_gu[ns]\|Compiling entry" | grep -o "Used [0-9]* registers.*spill[^,]*\|[0-9]* bytes spill stores" | head -4
done
for cfg in ${1:-1:8:1}; do echo "== UNROLL:MINB:LDU $cfg"; flock /root/gpu.lock ./roofline_$cfg $2 | grep -v "^SMs"; done
