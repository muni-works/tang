#!/usr/bin/env python3
"""One decode step's nsys trace (nsys_kernels.sh) grouped by what the kernels move, each group
against its analytic HBM floor at peak bandwidth:

  moe      routed + shared experts, routing, SwiGLU     48 · (U(N) · expert_MB + shared_MB)
  dense    the projection GEMVs / int8 GEMMs + head     non-expert weights − shared
  gdn      the GDN conv / state step kernels            36 · N · 2 · state_MB  (read + write)
  attn     attention part / merge / prep                12 · N · ctx · kv_bytes_per_token
  small    everything under 5 µs a launch (norms, quantizers, copies, argmax)   no floor

  step_groups.py BW N CSV WALL_MS [--model-info expert_mb=1.909 rest_mb=2359 shared_mb=3.34
                 state_mb=2.097 ctx=1380 kvtok=2048 e=512 k=10]
"""
import csv, re, sys
bw, n, path, wall = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
p = dict(expert_mb=1.909, rest_mb=2359, shared_mb=3.34, state_mb=2.097, ctx=1380, kvtok=2048, e=512, k=10,
         moe_layers=48, gdn_layers=36, attn_layers=12)
for kv in sys.argv[5:]:
    a, b = kv.split("="); p[a] = float(b)
u = p["e"] * (1 - (1 - p["k"] / p["e"]) ** n)
floor_mb = dict(
    moe=p["moe_layers"] * (u * p["expert_mb"] + p["shared_mb"]),
    dense=p["rest_mb"] - p["moe_layers"] * p["shared_mb"],
    gdn=p["gdn_layers"] * n * 2 * p["state_mb"],
    attn=p["attn_layers"] * n * p["ctx"] * p["kvtok"] / 1e6,
)
def group(name, us):
    if us < 5: return "small"
    if re.search(r"moe_(gu|down|plan)|ex_|sh_|silu|route|sort_|TensorId|weights", name): return "moe"
    if re.search(r"gemv|mma_q8x|argmax", name): return "dense"
    if "gdn" in name: return "gdn"
    if "attn" in name: return "attn"
    return "small"
T = {}
rows = [r for r in csv.DictReader(open(path)) if r.get("Name")]
ev = []
per = {}
for r in rows:
    nm = re.sub(r"\(.*$", "", r["Name"].strip('"')).split("::")[-1]
    d = float(r["Duration (ns)"]) / 1e3
    per.setdefault(nm, []).append(d)
    ev.append((float(r["Start (ns)"]), float(r["Duration (ns)"])))
for nm, ds in per.items():
    g = group(nm, sum(ds) / len(ds))
    T[g] = T.get(g, 0) + sum(ds) / 1e3
ev.sort()
idle = sum(max(0, ev[i][0] - ev[i-1][0] - ev[i-1][1]) for i in range(1, len(ev))) / 1e6
print(f"| group | ms | share of wall | floor MB | floor ms | light |")
print("|---|---|---|---|---|---|")
tot_floor = 0
for g in ["moe", "dense", "gdn", "attn", "small"]:
    t = T.get(g, 0)
    if g in floor_mb:
        fm = floor_mb[g]; fms = fm * 1e6 / bw * 1e3; tot_floor += fms
        print(f"| {g} | {t:.3f} | {100*t/wall:.0f}% | {fm:.0f} | {fms:.3f} | {fms/t if t else 0:.2f} |")
    else:
        print(f"| {g} | {t:.3f} | {100*t/wall:.0f}% | – | – | – |")
print(f"| idle between kernels | {idle:.3f} | {100*idle/wall:.0f}% | | | |")
print(f"| **step** | **{wall:.3f}** | | {sum(floor_mb.values()):.0f} | {tot_floor:.3f} | **{tot_floor/wall:.2f}** |")
print(f"\n{len(ev)} launches; distinct experts a layer U({n}) = {u:.1f}")
