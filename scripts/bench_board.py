#!/usr/bin/env python3
"""Board rows from derived-race outputs: one JSON line per (model, context).

    bench_board.py parse <race OUT dir or log> --model qwen3-8b [--branch B] [--date D]
        print rows for one run (bench-all.sh appends them to board.jsonl)
    bench_board.py backfill [--board bench/results/board.jsonl] [--race-artifacts DIR]
        rebuild the backfilled rows from the old results files; rows from
        bench-all.sh runs (source "bench-all") are kept

Row fields: date, commit, branch, gpu, model, context, hand_ms, hand_tok_s,
derived_ms, derived_tok_s, ids_matched (true/false/null), ids (e.g. "64/64"),
kernels (derived step, null when the run didn't print it), source.
Optional: kind ("step" when absent, "window": a k-token derived window,
"spec": prompt-lookup speculation end to end), window (k), config (the
non-default knobs the run used, e.g. "TANG_STAGED_MAP=0"; a knob whose
default has since changed is spelled out with "(default then)"), prompt,
greedy_tok_s and acceptance (spec rows), note.

    bench_board.py add --model M --gpu G --commit C --set derived_ms=2.42 ...
        print one hand-entered row (a result only a PR body records)
Units: ms per decode step (device time) and tok/s. Hand rows of the coder
engine are flash-bench wall tok/s; hand_ctx says what context it ran at.
"""
import argparse
import datetime
import json
import os
import re
import sys

LEN_RE = re.compile(
    r"decode step at len (\d+) \((hand|derived)\): ([\d.]+) ms device \(([\d.]+) tok/s\)")
DER_RE = re.compile(r"derived decode step at len (\d+): ([\d.]+) ms device \(([\d.]+) tok/s\)")
WIN_RE = re.compile(
    r"window of (\d+) at len (\d+) \(derived\): ([\d.]+) ms device \(([\d.]+) tok/s\)")
GREEDY_RE = re.compile(r"^\s*decode (\d+) tokens in [\d.]+s: ([\d.]+) tok/s end to end", re.M)
SPEC_RE = re.compile(r"spec k=(\d+): \d+ tokens in [\d.]+s: ([\d.]+) tok/s end to end;.*?"
                     r"acceptance (\d+)%")
SPEC_EQ_RE = re.compile(r"spec k=\d+: first (\d+)/(\d+) tokens equal")
HAND_RE = re.compile(r"hand \(ctx (\d+)\):\s+decode\s+([\d.]+) tok/s")


def read(path):
    if os.path.isdir(path):
        parts = []
        for name in ("summary.txt", "commit.txt", "gpu.txt", "derived.txt",
                     "model-hand.txt", "model-derived.txt"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                parts.append(f"### {name}\n" + open(p, errors="replace").read())
        return "\n".join(parts)
    text = open(path, errors="replace").read()
    d = os.path.dirname(path)
    head = ""
    c, g = os.path.join(d, "commit.txt"), os.path.join(d, "gpu.txt")
    if "commit: " not in text and os.path.exists(c):
        head += "commit: " + open(c, errors="replace").read().strip() + "\n"
    if "GPU: " not in text and os.path.exists(g):
        lines = [l for l in open(g, errors="replace").read().splitlines() if l.startswith("NVIDIA")]
        if lines:
            head += "GPU: " + lines[0] + "\n"
    return head + text


def meta(text):
    gpu = commit = None
    m = re.search(r"GPU: ([^,\n]+)", text) or re.search(r"^(NVIDIA [^,\n]+),", text, re.M) \
        or re.search(r"(NVIDIA [\w -]+?) \(\d+ GB\)", text)
    if m:
        gpu = m.group(1).strip()
    m = re.search(r"commit: (\S+)", text)
    if m and m.group(1) != "no":
        commit = m.group(1)
    return gpu, commit


def ids(text):
    m = re.search(r"first (\d+) of (\d+) identical", text)
    if m:
        return m.group(1) == m.group(2), f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"(\d+)/(\d+) ids identical", text)
    if m:
        return m.group(1) == m.group(2), f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"greedy ids(?:, derived vs hand)?: (identical|differ)(?: \((\d+) tokens\))?", text)
    if m:
        n = m.group(2)
        return m.group(1) == "identical", (f"{n}/{n}" if n and m.group(1) == "identical" else None)
    return None, None


def kernels(text):
    m = re.search(r"derived step: (\d+) kernels", text)
    if m:
        return int(m.group(1))
    # qwen_decode prints "step seq 1 ...: N kernels" per run; the derived run is the second
    found = re.findall(r"step seq 1 .*?: (\d+) kernels", text)
    if len(found) >= 2:
        return int(found[1])
    m = re.search(r"over (\d+) kernels", text)
    return int(m.group(1)) if m else None


def tok(ms):
    return round(1000.0 / ms, 1) if ms else None


def parse(text, model):
    """{context: row} for one run's text."""
    rows = {}

    def row(ctx):
        return rows.setdefault(ctx, {"context": ctx})

    # qwen: "decode step at len N (hand|derived)"; the first of each wins
    # (derived-last prints "(derived)" too, after derived)
    for ctx, kind, ms, ts in LEN_RE.findall(text):
        r = row(int(ctx))
        if f"{kind}_ms" not in r:
            r[f"{kind}_ms"], r[f"{kind}_tok_s"] = float(ms), float(ts)
    # coder: "derived decode step at len N" and "hand (ctx N): decode X tok/s"
    for ctx, ms, ts in DER_RE.findall(text):
        r = row(int(ctx))
        r.setdefault("derived_ms", float(ms))
        r.setdefault("derived_tok_s", float(ts))
    hands = HAND_RE.findall(text)
    ctxs = sorted(rows)
    for hctx, ts in hands:
        hctx = int(hctx)
        # the short-prompt hand bench goes with the smallest bench context
        target = hctx if hctx in rows else (ctxs[0] if ctxs and hctx < ctxs[0] else hctx)
        r = row(target)
        if "hand_tok_s" not in r:
            r["hand_tok_s"] = float(ts)
            r["hand_ms"] = round(1000.0 / float(ts), 3)
            if target != hctx:
                r["hand_ctx"] = hctx
    # the coder-next real-size write-up: a two-column table
    m = re.search(r"ctx (\d+)\s+ctx (\d+)\s*\n\s*derived\s+([\d.]+) ms \(([\d.]+) tok/s\)\s+([\d.]+) ms \(([\d.]+) tok/s\)\s*\n"
                  r"\s*hand engine\s+~?([\d.]+) ms \(([\d.]+) tok/s\)\s+~?([\d.]+) ms \(([\d.]+) tok/s\)", text)
    if m and not rows:
        g = m.groups()
        for i, ctx in enumerate((g[0], g[1])):
            r = row(int(ctx))
            r["derived_ms"], r["derived_tok_s"] = float(g[2 + 2 * i]), float(g[3 + 2 * i])
            r["hand_ms"], r["hand_tok_s"] = float(g[6 + 2 * i]), float(g[7 + 2 * i])
    gpu, commit = meta(text)
    matched, n = ids(text)
    k = kernels(text)
    out = []
    for ctx in sorted(rows):
        r = rows[ctx]
        out.append({
            "commit": commit, "gpu": gpu, "model": model, "context": ctx,
            "hand_ms": r.get("hand_ms"), "hand_tok_s": r.get("hand_tok_s"),
            "derived_ms": r.get("derived_ms"), "derived_tok_s": r.get("derived_tok_s"),
            "ids_matched": matched, "ids": n, "kernels": k,
            **({"hand_ctx": r["hand_ctx"]} if "hand_ctx" in r else {}),
        })
    base = {"commit": commit, "gpu": gpu, "model": model}
    seen = set()
    for w, ctx, ms, ts in WIN_RE.findall(text):
        if (w, ctx) in seen:
            continue
        seen.add((w, ctx))
        out.append(dict(base, kind="window", window=int(w), context=int(ctx),
                        derived_ms=float(ms), derived_tok_s=float(ts), kernels=None))
    spec = SPEC_RE.search(text)
    if spec:
        g = GREEDY_RE.search(text)
        eq = SPEC_EQ_RE.search(text)
        out.append(dict(base, kind="spec", window=int(spec.group(1)), context=None,
                        derived_tok_s=float(spec.group(2)), acceptance=int(spec.group(3)) / 100,
                        greedy_tok_s=float(g.group(2)) if g else None,
                        ids_matched=(eq.group(1) == eq.group(2)) if eq else None,
                        ids=f"{eq.group(1)}/{eq.group(2)}" if eq else None))
    return out


KEYS = ["date", "commit", "branch", "gpu", "model", "kind", "window", "context", "config",
        "prompt", "hand_ms", "hand_tok_s", "derived_ms", "derived_tok_s", "greedy_tok_s",
        "acceptance", "ids_matched", "ids", "kernels", "hand_ctx", "note", "source"]


def stamp(rows, date, branch, source, config=None, prompt=None):
    keys = KEYS
    done = []
    for r in rows:
        r = dict(r, date=date, branch=r.get("branch") or branch, source=source)
        if config:
            r.setdefault("config", config)
        if prompt:
            r.setdefault("prompt", prompt)
        done.append({k: r[k] for k in keys if k in r})
    return done


def mtime_date(path):
    p = os.path.join(path, "summary.txt") if os.path.isdir(path) else path
    return datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()


# race dirs named for the knob they ran with; the default then is spelled out
# where it has since changed (staged maps and cokernels are opt-in on main)
CONFIG = {
    "cdr-037e3984": "TANG_STAGED_MAP=1 (default then)",
    "cdr-037e3984-staged0": "TANG_STAGED_MAP=0",
    "cdr-multiout": "TANG_STAGED_MAP=1 TANG_COKERNEL=1 (defaults then)",
    "cdr-multiout-cokernel0": "TANG_STAGED_MAP=1 (default then) TANG_COKERNEL=0",
    "cdr-multiout-rules-a": "TANG_STAGED_MAP=1 TANG_COKERNEL=1 (defaults then) TANG_COKERNEL_RULES=a",
}


def file_config(name):
    """bench-gemv8-w4.txt -> TANG_WINDOW_GEMV=8; knob-K=V.txt -> K=V"""
    m = re.match(r"bench-gemv(\d+)-w\d+\.txt$", name)
    if m:
        # 8 (the tiled window GEMV) is the default
        return None if m.group(1) == "8" else f"TANG_WINDOW_GEMV={m.group(1)}"
    m = re.match(r"knob-(.+)\.txt$", name)
    return m.group(1) if m else None


def backfill_sources(repo, artifacts):
    """(path, model, date or None, branch, commit override[, config, prompt])"""
    # oldest first: the board orders runs by date, then by line
    dr = os.path.join(repo, "bench/results/derived-race")
    a = lambda *x: os.path.join(artifacts or "", *x)
    have_art = bool(artifacts) and os.path.isdir(artifacts)
    src = []
    if have_art:
        # the coder-next first correct run (5fc5ceb7, rsynced: no commit in the file)
        src.append((a("coder-race-3.log"), "coder-next", None, "claude/derive-integrate", "5fc5ceb7"))
        # the Qwen3-8B half of derive-race-all (rsynced copy, no commit)
        src.append((a("derive-race-all-6000.log"), "qwen3-8b", None, "claude/derive-integrate", None))
    src += [
        (os.path.join(dr, "rtx-pro-6000.txt"), "qwen3-8b", None, "claude/decode-from-prefill", None),
        (os.path.join(dr, "coder-next-realsize-rtx-pro-6000.txt"), "coder-next", "2026-11-11",
         "claude/derive-integrate", None),
    ]
    for name in sorted(os.listdir(dr)) if os.path.isdir(dr) else []:
        p = os.path.join(dr, name)
        if name.endswith(".txt") and p not in [s[0] for s in src]:
            src.append((p, "coder-next" if "coder" in name else "qwen3-8b", None, None, None))
    if have_art and os.path.isdir(a("abd-6000")):
        cdr = [a("abd-6000", n) for n in os.listdir(a("abd-6000")) if n.startswith("cdr-")]
        cdr = [p for p in cdr if os.path.exists(os.path.join(p, "summary.txt"))]
        for p in sorted(cdr, key=lambda p: os.path.getmtime(os.path.join(p, "summary.txt"))):
            branch = "claude/multi-out" if "multiout" in p else "claude/fuse-gdn-norm" \
                if "037e3984" in p or "5f930c7c" in p else "claude/derive-integrate"
            src.append((p, "coder-next", None, branch, None, CONFIG.get(os.path.basename(p))))
        # prompt lookup: one file a bench config or a prompt
        for run in ("prompt-lookup", "prompt-lookup-2"):
            d = a("abd-6000", run)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if name.startswith(("bench-", "knob-")):
                    src.append((p, "qwen3-8b", None, "claude/prompt-lookup", None,
                                file_config(name)))
                m = re.match(r"spec-(.+)-k\d+\.txt$", name)
                if m:
                    src.append((p, "qwen3-8b", None, "claude/prompt-lookup", None, None,
                                m.group(1)))
    return src


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("path")
    p.add_argument("--model", required=True)
    p.add_argument("--branch")
    p.add_argument("--date")
    p.add_argument("--source", default="bench-all")
    p.add_argument("--config", help="non-default knobs, e.g. TANG_COKERNEL=1")
    p.add_argument("--prompt")
    p.add_argument("--gpu", help="GPU name for rows whose output doesn't say")
    ad = sub.add_parser("add")
    for k in ("model", "gpu", "commit"):
        ad.add_argument(f"--{k}", required=True)
    ad.add_argument("--branch")
    ad.add_argument("--date")
    ad.add_argument("--source", default="manual")
    ad.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="any row field; numbers, true/false and null are parsed")
    b = sub.add_parser("backfill")
    b.add_argument("--repo", default=".")
    b.add_argument("--board", default="bench/results/board.jsonl")
    b.add_argument("--race-artifacts", default=os.path.expanduser("~/Developer/tang-race/artifacts"))
    args = ap.parse_args()

    if args.cmd == "parse":
        rows = parse(read(args.path), args.model)
        for r in rows:
            r["gpu"] = r.get("gpu") or args.gpu
        date = args.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        for r in stamp(rows, date, args.branch, args.source, args.config, args.prompt):
            print(json.dumps(r))
        return 0 if rows else 1

    if args.cmd == "add":
        r = {"model": args.model, "gpu": args.gpu, "commit": args.commit}
        for kv in args.set:
            k, v = kv.split("=", 1)
            try:
                r[k] = json.loads(v)
            except ValueError:
                r[k] = v
        date = args.date or datetime.date.today().isoformat()
        print(json.dumps(stamp([r], date, args.branch, args.source)[0]))
        return 0

    kept = []
    if os.path.exists(args.board):
        for line in open(args.board):
            line = line.strip()
            if line and not json.loads(line).get("source", "").startswith("backfill"):
                kept.append(line)
    new = []
    for path, model, date, branch, commit, *rest in backfill_sources(args.repo, args.race_artifacts):
        config, prompt = (rest + [None, None])[:2]
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        rows = parse(read(path), model)
        name = os.path.basename(path)
        # every bench file reruns the plain decode step: keep one copy (a
        # knob file keeps its own: the knob changes the step too)
        if name.startswith("bench-") and name != "bench-gemv8-w4.txt":
            rows = [r for r in rows if r.get("kind")]
        if commit:
            for r in rows:
                r["commit"] = r["commit"] or commit
        rel = os.path.relpath(path, args.repo) if path.startswith(os.path.abspath(args.repo)) else os.path.basename(path)
        if os.path.dirname(os.path.dirname(path)).endswith("abd-6000") and not os.path.isdir(path):
            rel = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        rows = stamp(rows, date or mtime_date(path), branch, f"backfill:{rel}", config, prompt)
        print(f"{len(rows)} rows: {path}", file=sys.stderr)
        new += [json.dumps(r) for r in rows]
    os.makedirs(os.path.dirname(args.board) or ".", exist_ok=True)
    with open(args.board, "w") as f:
        for line in new + kept:
            f.write(line + "\n")
    print(f"{args.board}: {len(new)} backfilled + {len(kept)} kept", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
