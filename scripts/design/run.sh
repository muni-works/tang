#!/bin/bash
# Pod entrypoint: prep data, smoke-test every mixer, then train them one per GPU.
# MIXERS and TOKENS from env. Results: RES lines in the log, and a gzip|base64
# dump of all jsonl at the end.
cd /work
pip install -q tiktoken pyarrow flash-linear-attention 2>&1 | tail -2
python -c "import torch, fla; print('torch', torch.__version__, 'fla ok')"
python prep.py /data ${SHARDS:-2} || exit 1
MIX=(${MIXERS:-softmax mamba2 gdn forget logdist sigmoid forget_sigmoid softcap_forget})
ok=()
pids=()
for i in "${!MIX[@]}"; do
  m=${MIX[$i]}
  CUDA_VISIBLE_DEVICES=$i timeout 900 python train.py $m --tokens 4e6 --eval_every 30 > smoke_$m.log 2>&1 &
  pids+=($!)
done
for i in "${!MIX[@]}"; do
  m=${MIX[$i]}
  if wait ${pids[$i]}; then ok+=($m); echo "SMOKE OK $m $(grep throughput smoke_$m.log)"; else echo "SMOKE FAIL $m"; tail -25 smoke_$m.log; fi
done
pids=()
for i in "${!ok[@]}"; do
  m=${ok[$i]}
  CUDA_VISIBLE_DEVICES=$i python train.py $m --tokens ${TOKENS:-1e9} --out res_$m.jsonl > full_$m.log 2>&1 &
  pids+=($!)
done
( tail -q -F full_*.log 2>/dev/null | grep --line-buffered -E "RES|Error|error" ) &
for p in "${pids[@]}"; do wait $p; done
for m in "${ok[@]}"; do echo "DONE $m"; tail -3 full_$m.log; done
echo "DUMP_BEGIN"; cat res_*.jsonl | gzip | base64 -w0; echo; echo "DUMP_END"
sleep infinity
