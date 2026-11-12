#!/usr/bin/env bash
# `tang-llm serve` on Qwen3-Coder-Next: the hand engine vs TANG_CODER_ENGINE=derived,
# same binary (tang-bench's coder_serve), real OpenAI-API requests on localhost.
#
#   1. build coder_serve (cuda, gguf, tang-llm/cuda)
#   2. for each engine: serve --batch 4, then
#        C=1: each prompt alone, greedy, streamed (TTFT, decode tok/s, text)
#        C=4: the same prompts at once
#        derived only: one sampled prompt twice with the same seed (reproducible?)
#   3. greedy texts compared hand vs derived (C=1 and C=4), summary.json
#
# Run from the repo root (gpu-run.sh race:coder-next:scripts/coder-serve-check.sh).
# Env: MODEL_DIR [/root/models/coder-next], OUT [/root/coder-serve-check], CTX [8192],
# MAX_TOKENS [128], LONG_WORDS [1500: long-prompt TTFT], ENGINES ["hand derived"].
set -euo pipefail
REPO=${REPO:-$PWD}
MODEL_DIR=${MODEL_DIR:-/root/models/coder-next}
OUT=${OUT:-/root/coder-serve-check}
CTX=${CTX:-8192}
MAX_TOKENS=${MAX_TOKENS:-128}
ENGINES=${ENGINES:-hand derived}
GGUF=$(ls "$MODEL_DIR"/*.gguf | head -1)
mkdir -p "$OUT"
cd "$REPO"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
git log --oneline -1 | tee "$OUT/commit.txt"

echo "=== build"
cargo build --release -p tang-bench --features coder-serve --example coder_serve 2>&1 | tail -3
BIN=${CARGO_TARGET_DIR:-$REPO/target}/release/examples/coder_serve

cat > "$OUT/client.py" <<'EOF'
import json, sys, time, threading, urllib.request
port, mode, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
MAX = int(sys.argv[4])
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain what a Rust lifetime is in two sentences.",
    "Write a bash one-liner that counts lines in all .rs files under the current directory.",
    "What does the HTTP status code 429 mean, and how should a client react?",
]
def req(prompt, temperature=0.0, seed=None):
    body = {"model": "coder", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body["top_p"] = 0.95; body["top_k"] = 40
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time(); first = None; text = ""; chunks = 0; usage = None
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if first is None:
                        first = time.time()
                    text += piece; chunks += 1
    t1 = time.time()
    n = (usage or {}).get("completion_tokens") or chunks
    first = first or t1
    dec = t1 - first
    return {"text": text, "tokens": n, "ttft_ms": (first - t0) * 1e3, "total_ms": (t1 - t0) * 1e3,
            "decode_tok_s": (n - 1) / dec if dec > 0 and n > 1 else None}
res = {"c1": [], "c4": []}
req(PROMPTS[0])  # warm
for p in PROMPTS:
    res["c1"].append(req(p))
out4 = [None] * len(PROMPTS)
def run(i): out4[i] = req(PROMPTS[i])
t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
[t.start() for t in th]; [t.join() for t in th]
res["c4"] = out4
res["c4_wall_ms"] = (time.time() - t0) * 1e3
res["c4_agg_tok_s"] = sum(x["tokens"] for x in out4) / (res["c4_wall_ms"] / 1e3)
# long prompt TTFT (LONG_WORDS words of code-ish text, ~1.3 tokens a word; 0: skip)
import os
LW = int(os.environ.get("LONG_WORDS", "1500"))
if LW:
    words = ("fn step(x: &mut [f32], k: usize) -> f32 { let s = x.iter().sum::<f32>(); s / k as f32 } ").split()
    long_p = "Summarize what this code does:\n" + " ".join(words[i % len(words)] for i in range(LW))
    MAX0 = MAX; MAX = 8
    req(long_p + " warm")
    res["long"] = [req(long_p) for _ in range(2)]
    MAX = MAX0
if mode == "derived":
    a = req(PROMPTS[1], 0.8, 7); b = req(PROMPTS[1], 0.8, 7); c = req(PROMPTS[1], 0.8, 8)
    res["sampled"] = {"same_seed_equal": a["text"] == b["text"], "other_seed_differs": a["text"] != c["text"],
                      "a": a["text"][:200], "c": c["text"][:200]}
json.dump(res, open(out, "w"), indent=1)
EOF

port=8931
for eng in $ENGINES; do
  echo "=== serve: $eng"
  if [ "$eng" = derived ]; then export TANG_CODER_ENGINE=derived; else unset TANG_CODER_ENGINE; fi
  # bitwise-greedy windows on the hand engine so its batched output is comparable
  TANG_SERVE_DETERMINISTIC=1 TANG_SERVE_WIDE=0 "$BIN" "$GGUF" --port $port --ctx "$CTX" --batch 4 \
    --no-suffix --snapshots 0 > "$OUT/serve-$eng.log" 2>&1 &
  pid=$!
  t0=$(date +%s)
  until curl -sf -m 2 http://127.0.0.1:$port/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server died"; tail -30 "$OUT/serve-$eng.log"; exit 1; }
    sleep 5
  done
  echo "ready in $(( $(date +%s) - t0 )) s"
  python3 "$OUT/client.py" $port $eng "$OUT/$eng.json" "$MAX_TOKENS" || { tail -40 "$OUT/serve-$eng.log"; kill $pid; exit 1; }
  kill $pid; wait $pid 2>/dev/null || true
  grep -E 'coder derived|engine ready|TANG_CODER' "$OUT/serve-$eng.log" || true
  port=$((port + 1))
done

python3 - "$OUT" <<'EOF'
import json, sys, os, statistics as st
o = sys.argv[1]; r = {e: json.load(open(f"{o}/{e}.json")) for e in ("hand", "derived") if os.path.exists(f"{o}/{e}.json")}
s = {}
for e, x in r.items():
    s[e] = {
      "c1_ttft_ms_median": st.median(y["ttft_ms"] for y in x["c1"]),
      "c1_decode_tok_s_median": st.median(y["decode_tok_s"] for y in x["c1"] if y["decode_tok_s"]),
      "c1_ms_per_token": 1e3 / st.median(y["decode_tok_s"] for y in x["c1"] if y["decode_tok_s"]),
      "c4_per_stream_tok_s_median": st.median(y["decode_tok_s"] for y in x["c4"] if y["decode_tok_s"]),
      "c4_ttft_ms_median": st.median(y["ttft_ms"] for y in x["c4"]),
      "c4_aggregate_tok_s": x["c4_agg_tok_s"],
    }
    if "long" in x: s[e]["long_ttft_ms_median"] = st.median(y["ttft_ms"] for y in x["long"])
    if "sampled" in x: s[e]["sampled"] = {k: v for k, v in x["sampled"].items() if k.endswith(("equal", "differs"))}
if len(r) == 2:
    h, d = r["hand"], r["derived"]
    s["greedy_equal_c1"] = [a["text"] == b["text"] for a, b in zip(h["c1"], d["c1"])]
    s["greedy_equal_c4"] = [a["text"] == b["text"] for a, b in zip(h["c4"], d["c4"])]
    s["derived_c1_equals_c4"] = [a["text"] == b["text"] for a, b in zip(d["c1"], d["c4"])]
    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y: break
            n += 1
        return n
    s["common_prefix_chars_c1"] = [(common(a["text"], b["text"]), len(a["text"])) for a, b in zip(h["c1"], d["c1"])]
json.dump(s, open(f"{o}/summary.json", "w"), indent=1)
print(json.dumps(s, indent=1))
EOF
