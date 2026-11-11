"""Summarize docs/design/runs/*.jsonl: final val loss (equal tokens), val loss
at the softmax baseline's wall-clock (equal time), throughput, params.
usage: python analyze.py docs/design/runs [plot.png]"""
import glob, json, sys
from collections import defaultdict

runs = defaultdict(list)
thr = {}
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if "throughput" in d:
            thr[d["mixer"]] = d["throughput"]
        elif "val" in d:
            runs[d["mixer"]].append(d)

for m in runs:
    runs[m].sort(key=lambda d: d["step"])


def at_wall(pts, w):
    """val loss at wall-clock w (linear interp; None past the end)."""
    for a, b in zip(pts, pts[1:]):
        if a["wall"] <= w <= b["wall"]:
            t = (w - a["wall"]) / (b["wall"] - a["wall"])
            return a["val"] + t * (b["val"] - a["val"])
    return None


base_wall = runs["softmax"][-1]["wall"]
base = runs["softmax"][-1]["val"]
rows = []
for m, pts in runs.items():
    fin = pts[-1]
    tok = fin["tokens"]
    # forward+backward FLOPs/token ~ 6N (dense) + 12*L*T*D attention-like term
    # (quadratic mixers only; reported, not used for ranking)
    rows.append((fin["val"], m, fin["params"], tok, thr.get(m, 0), fin["wall"], at_wall(pts, base_wall)))
rows.sort()
print("| mixer | params | tokens | val loss (1B tok) | Δ vs softmax | tok/s | wall s | val @ softmax wall |")
print("|---|---|---|---|---|---|---|---|")
for v, m, p, tok, t, w, vw in rows:
    vw = f"{vw:.3f}" if vw else "—"
    print(f"| {m} | {p/1e6:.1f}M | {tok/1e9:.2f}B | {v:.3f} | {v-base:+.3f} | {t/1e3:.0f}k | {w:.0f} | {vw} |")

if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, pts in sorted(runs.items()):
        pts = [d for d in pts if d["step"] >= 2000]
        ax[0].plot([d["tokens"] / 1e9 for d in pts], [d["val"] for d in pts], label=m)
        ax[1].plot([d["wall"] / 3600 for d in pts], [d["val"] for d in pts], label=m)
    ax[0].set_xlabel("tokens (B)"); ax[1].set_xlabel("wall-clock (h, RTX PRO 6000, torch)")
    for a in ax:
        a.set_ylabel("val loss"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=120)
