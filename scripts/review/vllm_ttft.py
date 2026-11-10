# vLLM time to first token on Qwen3-Coder-Next-FP8: prompt lengths of N tokens, one generated
# token, three different prompts each (no prefix cache in an offline LLM), after one warm-up run.
# Prints every run. Model load is reported apart and is not in TTFT.
import sys, time, random
from vllm import LLM, SamplingParams
model, src = sys.argv[1], open(sys.argv[2]).read()
t0 = time.time()
llm = LLM(model=model, max_model_len=64000, gpu_memory_utilization=0.92, enforce_eager=False)
print(f"load {time.time()-t0:.1f} s", flush=True)
import vllm; print("vllm", vllm.__version__, flush=True)
tok = llm.get_tokenizer()
base = tok.encode(src)
sp = SamplingParams(temperature=0, max_tokens=1)
rng = random.Random(7)
def prompt(n):
    # a different cyclic rotation each time, so no two prompts share a prefix
    s = rng.randrange(len(base))
    return {"prompt_token_ids": [(base[(s + i) % len(base)]) for i in range(n)]}
llm.generate([prompt(1000)], sp)
for n in (8000, 44000):
    for r in range(3):
        t = time.time(); o = llm.generate([prompt(n)], sp); dt = time.time() - t
        print(f"vllm ttft n={n} run {r}: {dt:.3f} s ({len(o[0].prompt_token_ids)} prompt tokens)", flush=True)
