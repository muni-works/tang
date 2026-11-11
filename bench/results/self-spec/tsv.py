"""Emit TSV: model, drafter, cost, windows, P1, and acc/win + tokens per full-model read at k=1,2,4,8."""
import json, sys
model = sys.argv[1]
cost, matches = {}, {}
for path in sys.argv[2:]:
    for line in open(path):
        r = json.loads(line)
        if "cost" in r:
            cost[r["drafter"]] = (r["cost"], r["layers"], r["bits"], r["own_kv"])
        else:
            matches.setdefault(r["drafter"], []).extend(r["match"])
for d, m in matches.items():
    c, nl, b, own = cost[d]
    cells = []
    for k in (1, 2, 4, 8):
        acc = sum(min(x, k) for x in m) / len(m)
        cells += [f"{acc:.2f}", f"{(acc + 1) / (1 + k * c):.2f}"]
    p1 = sum(x >= 1 for x in m) / len(m)
    print("\t".join([model, d, str(nl), str(b), "own" if own else "shared", f"{c:.3f}", str(len(m)), f"{p1:.2f}"] + cells))
