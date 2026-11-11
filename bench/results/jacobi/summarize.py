#!/usr/bin/env python3
"""Summarize jacobi_cpu TSV output: tokens per weight read (iteration) and
rows per finalized token, per method, split code vs prose.

    summarize.py q8b.tsv [q17b.tsv ...]
"""
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # toks iters rows secs exact-fail
    order = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        name, method, toks, iters, rows, secs, exact = f[:7]
        cat = "code" if name.startswith("code") else "prose"
        for c in (cat, "all"):
            a = agg[(method, c)]
            a[0] += int(toks)
            a[1] += int(iters)
            a[2] += int(rows)
            a[3] += float(secs)
            a[4] += exact != "1"
        if method not in order:
            order.append(method)
    print(f"## {path}\n")
    print("| method | code tok/iter | prose tok/iter | all tok/iter | rows/token (all) | wasted rows | exact |")
    print("|---|---|---|---|---|---|---|")
    for m in order:
        c, p, a = agg[(m, "code")], agg[(m, "prose")], agg[(m, "all")]
        ti = lambda x: x[0] / x[1] if x[1] else float("nan")
        waste = 1 - a[0] / a[2] if a[2] else 0
        ok = "yes" if a[4] == 0 else f"NO ({a[4]})"
        print(f"| {m} | {ti(c):.2f} | {ti(p):.2f} | {ti(a):.2f} | {a[2]/a[0]:.2f} | {waste:.0%} | {ok} |")
    print()
