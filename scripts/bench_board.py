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
    return open(path, errors="replace").read()


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
    return out


def stamp(rows, date, branch, source):
    keys = ["date", "commit", "branch", "gpu", "model", "context", "hand_ms", "hand_tok_s",
            "derived_ms", "derived_tok_s", "ids_matched", "ids", "kernels", "hand_ctx", "source"]
    done = []
    for r in rows:
        r = dict(r, date=date, branch=r.get("branch") or branch, source=source)
        done.append({k: r[k] for k in keys if k in r})
    return done


def mtime_date(path):
    p = os.path.join(path, "summary.txt") if os.path.isdir(path) else path
    return datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()


def backfill_sources(repo, artifacts):
    """(path, model, date or None, branch, commit override)"""
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
            src.append((p, "coder-next", None, "claude/derive-integrate", None))
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
    b = sub.add_parser("backfill")
    b.add_argument("--repo", default=".")
    b.add_argument("--board", default="bench/results/board.jsonl")
    b.add_argument("--race-artifacts", default=os.path.expanduser("~/Developer/tang-race/artifacts"))
    args = ap.parse_args()

    if args.cmd == "parse":
        rows = parse(read(args.path), args.model)
        date = args.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        for r in stamp(rows, date, args.branch, args.source):
            print(json.dumps(r))
        return 0 if rows else 1

    kept = []
    if os.path.exists(args.board):
        for line in open(args.board):
            line = line.strip()
            if line and not json.loads(line).get("source", "").startswith("backfill"):
                kept.append(line)
    new = []
    for path, model, date, branch, commit in backfill_sources(args.repo, args.race_artifacts):
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        rows = parse(read(path), model)
        if commit:
            for r in rows:
                r["commit"] = r["commit"] or commit
        rel = os.path.relpath(path, args.repo) if path.startswith(os.path.abspath(args.repo)) else os.path.basename(path)
        rows = stamp(rows, date or mtime_date(path), branch, f"backfill:{rel}")
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
