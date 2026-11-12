#!/usr/bin/env python3
"""Agent-like trace against an OpenAI-compatible server (tang-llm serve), for prefix-cache
hit rate and TTFT.

SESSIONS conversations share one tools/system block (SYS_TOK words); each opens with its own
FIRST_TOK-word task, then runs TURNS turns: the model answers (MAX_TOKENS), a TOOL_TOK-word
"tool result" comes back as the next user message. At most CONC requests are in flight; with
SESSIONS > the server's batch, parked slots get evicted between a session's turns, which is
what spill-to-host / rewind are for. A session's turn REWIND_EVERY (0: never) edits its last
user message instead of appending, like an agent retrying a step (the prompt diverges inside
the conversation).

Cross-user sharing (all off by default, the #249 trace): TENANTS > 0 sends session i as OpenAI
`user` "user-{i % TENANTS}"; FILES > 0 opens each first message with one of FILES popular
FILE_TOK-word files (Zipf-ish: file k with weight 1/(k+1)), like an agent's AGENTS.md or a
README read first, before its own task; FORK_EVERY > 0 makes every FORK_EVERY-th session a
subagent of the session before it (same user, same first message, its own tool results).
SEED_OFFSET shifts the per-session text (0: the same sessions every run, which a restart pass
uses to replay returning users).

Prints and writes ($OUT/replay-<label>.json) per-request rows and a summary: hit rate (cached /
prompt tokens, from the server's usage), TTFT by turn, aggregate output tok/s.
"""
import json, os, random, statistics, sys, threading, time, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8931")
SESSIONS = int(os.environ.get("SESSIONS", 20))
TURNS = int(os.environ.get("TURNS", 6))
CONC = int(os.environ.get("CONC", 16))
SYS_TOK = int(os.environ.get("SYS_TOK", 3000))
FIRST_TOK = int(os.environ.get("FIRST_TOK", 800))
TOOL_TOK = int(os.environ.get("TOOL_TOK", 700))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 48))
REWIND_EVERY = int(os.environ.get("REWIND_EVERY", 4))
TENANTS = int(os.environ.get("TENANTS", 0))
FILES = int(os.environ.get("FILES", 0))
FILE_TOK = int(os.environ.get("FILE_TOK", 2000))
FORK_EVERY = int(os.environ.get("FORK_EVERY", 0))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", 0))
LABEL = os.environ.get("LABEL", "run")
OUT = os.environ.get("OUT", ".")

WORDS = ("file function return value error test module import class struct impl trait async "
         "await buffer index token cache slot state layer kernel prefix session request parse "
         "build check config path line column string number vector matrix tensor shape batch "
         "decode prefill window memory device host copy write read open close lock queue").split()


def text(rng, n):
    lines, line = [], []
    for _ in range(n):
        line.append(rng.choice(WORDS))
        if len(line) == 12:
            lines.append(" ".join(line))
            line = []
    return "\n".join(lines + [" ".join(line)])


SYSTEM = ("You are a coding agent. Tools:\n<IMPORTANT>\n" + text(random.Random(7), SYS_TOK)
          + "\n</IMPORTANT>\nWorking directory: /repo\n")


POPULAR = [f"<file path=\"docs/{k}.md\">\n" + text(random.Random(100 + k), FILE_TOK) + "\n</file>\n"
           for k in range(FILES)]


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, content, usage = None, [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    content.append(piece)
    return ttft, "".join(content), usage, time.time() - t0


rows, lock = [], threading.Lock()
sem = threading.Semaphore(CONC)


def first_message(i):
    """Session i's first user message: a popular file (if FILES), then its own task."""
    rng = random.Random(1000 + i + SEED_OFFSET)
    pre = ""
    if FILES:
        pre = rng.choices(POPULAR, weights=[1 / (k + 1) for k in range(FILES)])[0]
    return pre + f"Task {i}:\n" + text(rng, FIRST_TOK)


def session(i):
    parent = i - 1 if FORK_EVERY and i % FORK_EVERY == FORK_EVERY - 1 and i > 0 else i
    rng = random.Random(1000 + i + SEED_OFFSET + 77777 * (parent != i))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": first_message(parent)}]
    user = f"user-{parent % TENANTS}" if TENANTS else None
    for turn in range(TURNS):
        body = {"model": "coder", "messages": msgs, "max_tokens": MAX_TOKENS, "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 0,
                "prompt_cache_key": f"session-{i}"}
        if user:
            body["user"] = user
        with sem:
            ttft, content, usage, wall = post(body)
        u = usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        with lock:
            rows.append({"session": i, "fork": parent != i, "turn": turn, "ttft": ttft, "wall": wall,
                         "prompt": u.get("prompt_tokens", 0), "cached": cached,
                         "out": u.get("completion_tokens", 0), "t": time.time()})
        rewind = REWIND_EVERY and turn > 0 and turn % REWIND_EVERY == 0
        if rewind:
            # Retry the last step: replace the last assistant + user pair.
            msgs = msgs[:-2]
        msgs = msgs + [{"role": "assistant", "content": content},
                       {"role": "user", "content": f"Tool result {turn}:\n" + text(rng, TOOL_TOK)}]
        # Stagger, like an agent running its tool.
        time.sleep(rng.uniform(0.2, 1.5))


t0 = time.time()
ths = [threading.Thread(target=session, args=(i,)) for i in range(SESSIONS)]
for k, t in enumerate(ths):
    t.start()
    time.sleep(0.3)
for t in ths:
    t.join()
wall = time.time() - t0

prompt = sum(r["prompt"] for r in rows)
cached = sum(r["cached"] for r in rows)
out = sum(r["out"] for r in rows)
ttfts = [r["ttft"] for r in rows if r["ttft"] is not None]
later = [r["ttft"] for r in rows if r["ttft"] is not None and r["turn"] > 0]
first = [r["ttft"] for r in rows if r["ttft"] is not None and r["turn"] == 0]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


summary = {
    "label": LABEL, "sessions": SESSIONS, "turns": TURNS, "conc": CONC, "requests": len(rows),
    "prompt_tokens": prompt, "cached_tokens": cached,
    "hit_rate": round(cached / max(prompt, 1), 4),
    "ttft_p50_s": pct(ttfts, 0.5), "ttft_p90_s": pct(ttfts, 0.9),
    "ttft_first_turn_p50_s": pct(first, 0.5),
    "ttft_later_turns_p50_s": pct(later, 0.5), "ttft_later_turns_p90_s": pct(later, 0.9),
    "ttft_later_turns_mean_s": statistics.mean(later) if later else None,
    "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
    "ttft_first_turn_mean_s": statistics.mean(first) if first else None,
    "wall_s": round(wall, 1), "output_tokens": out,
    "output_tok_s": round(out / wall, 1),
    "prefilled_tokens": prompt - cached,
}
os.makedirs(OUT, exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(OUT, f"replay-{LABEL}.json"), "w"))
print(json.dumps(summary))
