#!/usr/bin/env python3
"""Analytic HBM floor of one Coder-Next decode step: the weight bytes a step must read once.

  step_floor.py MODEL.gguf N [N...]      (needs `pip install gguf`)

Every non-expert tensor is read once a step (attention, GDN, norms, router, shared expert,
the output head), except the token embedding (one row a stream). Routed experts: each
distinct (layer, expert) touched by the N streams' top-k is read once; with independent
uniform routing the expected count a layer is E·(1 − (1 − k/E)^N). Real routing is
correlated, so this over-counts slightly (a lower floor would only make light fractions
larger). KV / GDN state traffic at ~1.4k context is small next to the weights and is left out.
"""
import sys
from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
rest = exp = 0
layers = set()
ne = k = None
for f in r.fields.values():
    if f.name.endswith(".expert_count"):
        ne = int(f.parts[-1][0])
    if f.name.endswith(".expert_used_count"):
        k = int(f.parts[-1][0])
for t in r.tensors:
    nb = int(t.n_bytes)
    if "_exps" in t.name:
        exp += nb
        layers.add(t.name.split(".")[1])
    elif t.name.startswith("token_embd"):
        continue
    else:
        rest += nb
nl = len(layers)
per_expert = exp / nl / ne
print(f"layers {nl}, experts {ne}, top-{k}; non-expert {rest/1e6:.0f} MB, one expert {per_expert/1e6:.3f} MB")
for n in map(int, sys.argv[2:]):
    u = ne * (1 - (1 - k / ne) ** n)
    mb = (rest + nl * u * per_expert) / 1e6
    print(f"N={n}: distinct experts a layer {u:.1f}, floor bytes {mb:.0f} MB")
