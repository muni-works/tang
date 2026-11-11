"""Summarize self_spec JSON lines: tokens per window and per full weight read."""
import json, sys
cost, matches = {}, {}
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if "cost" in r:
            cost[r["drafter"]] = (r["cost"], r["layers"], r["bits"], r["own_kv"])
        else:
            matches.setdefault(r["drafter"], []).extend(r["match"])
print("| drafter | layers | bits | KV | cost c | windows | " + " | ".join(
    f"k={k} acc/win | k={k} tok/read" for k in (2, 4, 8)) + " |")
print("|" + "---|" * 12)
rows = []
for d, m in matches.items():
    c, nl, b, own = cost[d]
    cells = []
    best = 0
    for k in (2, 4, 8):
        acc = sum(min(x, k) for x in m) / len(m)
        tpr = (acc + 1) / (1 + k * c)
        best = max(best, tpr)
        cells.append(f"{acc:.2f} | {tpr:.2f}")
    rows.append((best, f"| {d} | {nl} | {b} | {'own' if own else 'shared'} | {c:.3f} | {len(m)} | " + " | ".join(cells) + " |"))
for _, r in sorted(rows, reverse=True):
    print(r)
print()
print("first-draft acceptance (P[match>=1]) and P[match>=j]:")
for d, m in matches.items():
    print(d, " ".join(f"{sum(x >= j for x in m) / len(m):.2f}" for j in range(1, 9)))
