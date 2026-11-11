#!/bin/bash
# KDA-projection requant eval (NOTES 2026-10-27 kda-q4). Run on the H200 under flock:
#   flock /root/gpu.lock scripts/glm5-kda-q4-eval.sh <tag> <gguf-dir> [ppl|gen|all]
# ppl: llama.cpp llama-perplexity on bench/glm5-ppl/eval.txt (-c 512, first 6 chunks, experts on the CPU);
#      tag "orig" writes the KL base, others report KL vs it.
# gen: tang ids gate (-n 8) + 3 greedy -n 128 generations (code, prose, chat); FFN layers
#      $FFN_LAYERS (default 0-29) on the GPU with MOE=fast, the rest on the CPU.
set -u
tag=$1; dir=$2; what=${3:-all}
O=/root/w-q4/out; mkdir -p $O
G=$(ls $dir/*00001-of-*.gguf)
L=/root/w-q4/llama.cpp/build-cuda/bin
T=/root/w-q4/src/target/release/tang-llm
E=/root/w-q4/src/bench/glm5-ppl/eval.txt
if [[ $what != gen ]]; then
  if [[ $tag == orig ]]; then kl="--kl-divergence-base $O/kl-orig.bin"; else kl="--kl-divergence-base $O/kl-orig.bin --kl-divergence"; fi
  timeout 7200 $L/llama-perplexity -m $G -f $E -c 512 -b 512 --chunks 6 -ngl 99 -ot "exps=CPU" -t 48 $kl > $O/ppl-$tag.log 2>&1
  echo "ppl $tag rc=$?"; grep -E "Final estimate|Mean PPL|Mean    KLD|Same top|RMS Δp" $O/ppl-$tag.log
fi
if [[ $what != ppl ]]; then
  export TANG_REQUIRE_CUDA=1 TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_KDA_G5F=2 TANG_GLM5_MOE=fast TANG_GLM5_GPU_FFN_LAYERS=${FFN_LAYERS:-0-29}
  timeout 1800 $T flash-generate $G --prompt "Write a haiku about the sea." -n 8 --temp 0 > $O/ids-$tag.log 2>&1; echo "ids $tag rc=$?"
  i=0
  for p in "Write a Rust function that returns the n-th Fibonacci number using iteration." \
           "Describe a small harbor town on a winter morning." \
           "What are three tips for keeping a sourdough starter healthy?"; do
    timeout 3600 $T flash-generate $G --prompt "$p" -n 128 --temp 0 > $O/gen$i-$tag.log 2>&1; echo "gen$i $tag rc=$?"; i=$((i+1))
  done
fi
