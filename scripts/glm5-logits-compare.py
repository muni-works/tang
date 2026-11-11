#!/usr/bin/env python3
"""Compare two TANG_GLM5_LOGITS_DUMP runs (same prompt, -n N, --temp 0).

usage: glm5-logits-compare.py REF.logits REF.ids NEW.logits NEW.ids NVOCAB

*.ids: the run's stdout (the `ids:` line) or a bare id list. Logits row i is the step that
fed gen[i] and produced gen[i + 1]; rows are compared while the inputs agree (i < first
id divergence). Prints: first divergence, top-1 agreement, max |dlogit|, mean KL(ref||new).
"""
import sys
import numpy as np


def ids(p):
    for line in open(p):
        if line.startswith("ids:"):
            return [int(v) for v in line.split()[1:]]
    return [int(v) for v in open(p).read().split()]


ra, ri, na, ni, nv = sys.argv[1:6]
nv = int(nv)
a = np.fromfile(ra, dtype="<f4").reshape(-1, nv).astype(np.float64)
b = np.fromfile(na, dtype="<f4").reshape(-1, nv).astype(np.float64)
gi, gj = ids(ri), ids(ni)
n = min(len(gi), len(gj))
d = next((i for i in range(n) if gi[i] != gj[i]), n)
rows = min(len(a), len(b), d)  # row i fed gen[0..=i]: valid while i < d
same = sum(x == y for x, y in zip(gi, gj))
print(f"ids: ref {len(gi)} new {len(gj)}; first divergence at position {d if d < n else 'none'}; "
      f"positionwise id agreement {same}/{n}")
if rows == 0:
    sys.exit("no comparable rows")
a, b = a[:rows], b[:rows]
top = (a.argmax(1) == b.argmax(1)).mean()
dmax = np.abs(a - b).max()


def lsm(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


la, lb = lsm(a), lsm(b)
kl = (np.exp(la) * (la - lb)).sum(1)
print(f"rows compared {rows}: top-1 agreement {100 * top:.2f}%, max |dlogit| {dmax:.4g}, "
      f"mean KL(ref||new) {kl.mean():.3e} (max {kl.max():.3e})")
