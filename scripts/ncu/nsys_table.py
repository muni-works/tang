#!/usr/bin/env python3
"""nsys_kernels.sh CSV -> per-kernel table, like table.py, with DRAM bytes joined from an
ncu table.py JSON of the same workload on another GPU (`--bytes`; bytes moved are a property
of the kernel and shape, not the GPU, up to cache effects), or none. Also the idle gap between
consecutive kernels (launch / dependency bubbles) and, with --floor-mb, a whole-window floor.

  nsys_table.py CSV --bw 4.8e12 [--bytes other.json] [--title T] [--step-ms X --steps N]
                [--floor-mb MB] [--json out.json]
"""
import argparse, csv, json, re
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--bw", type=float, required=True)
ap.add_argument("--bytes"); ap.add_argument("--title", default="")
ap.add_argument("--step-ms", type=float); ap.add_argument("--steps", type=float, default=1)
ap.add_argument("--floor-mb", type=float, help="analytic bytes the whole window must move")
ap.add_argument("--launch-us", type=float, default=5.0); ap.add_argument("--json")
a = ap.parse_args()

def short(n):
    n = re.sub(r"\(.*$", "", n); n = re.sub(r"^void\s+", "", n); n = re.sub(r"<.*>$", "", n)
    return n.split("::")[-1]

rows = [r for r in csv.DictReader(open(a.csv)) if r.get("Name")]
mb = {}
if a.bytes:
    for k in json.load(open(a.bytes))["kernels"]:
        mb[k["kernel"]] = k["mb"]
K = OrderedDict(); ev = []
for r in rows:
    name = short(r["Name"].strip('"'))
    g = f"{r['GrdX']}x{r['BlkX']}" if r.get("GrdY") in ("1", "") else f"{r['GrdX']}x{r['GrdY']}x{r['BlkX']}"
    key = f"{name}@{r['GrdX']}x{r['BlkX']}" if name.startswith("main") else name
    s, d = float(r["Start (ns)"]), float(r["Duration (ns)"])
    ev.append((s, d))
    k = K.setdefault(key, {"n": 0, "t": 0.0, "grid": r["GrdX"], "block": r["BlkX"]})
    k["n"] += 1; k["t"] += d * 1e-9
ev.sort()
gap = sum(max(0.0, ev[i][0] - (ev[i-1][0] + ev[i-1][1])) for i in range(1, len(ev))) * 1e-9 if ev else 0
span = (ev[-1][0] + ev[-1][1] - ev[0][0]) * 1e-9 if ev else 0
out = []
for name, k in K.items():
    t = k["t"] / k["n"]
    m = mb.get(name)
    floor = m * 1e6 / a.bw if m is not None else None
    light = floor / t if floor is not None and t > 0 else None
    cls = "launch" if t * 1e6 < a.launch_us else ("traffic" if light is not None and light >= 0.5 else ("latency" if light is not None else "?"))
    out.append(dict(kernel=name, launches=k["n"], us=t * 1e6, total_ms=k["t"] * 1e3, mb=m,
                    floor_us=None if floor is None else floor * 1e6, light=light,
                    grid=k["grid"], block=k["block"], cls=cls))
out.sort(key=lambda r: -r["total_ms"])
tot = sum(r["total_ms"] for r in out)
if a.title: print(f"#### {a.title}\n")
print("| kernel | launches | µs each | total ms | share | MB each (other GPU's ncu) | floor µs | light | grid×block | bound |")
print("|---|---|---|---|---|---|---|---|---|---|")
f = lambda v, p: "–" if v is None else f"{v:.{p}f}"
for r in out[:40]:
    print(f"| `{r['kernel']}` | {r['launches']} | {r['us']:.1f} | {r['total_ms']:.3f} | {100*r['total_ms']/tot:.1f}% | {f(r['mb'],2)} | {f(r['floor_us'],1)} | {f(r['light'],2)} | {r['grid']}×{r['block']} | {r['cls']} |")
if len(out) > 40: print(f"| … {len(out)-40} more | | | {sum(r['total_ms'] for r in out[40:]):.3f} | | | | | | |")
by = {}
for r in out: by[r["cls"]] = by.get(r["cls"], 0) + r["total_ms"]
print(f"\nkernels {tot:.3f} ms over {len(ev)} launches; span {span*1e3:.3f} ms, idle between kernels {gap*1e3:.3f} ms ({100*gap/span if span else 0:.0f}% of span); by class: " + ", ".join(f"{c} {v:.3f} ms" for c, v in sorted(by.items(), key=lambda x: -x[1])))
if a.floor_mb:
    fl = a.floor_mb * 1e6 / a.bw * 1e3
    wall = a.step_ms * a.steps if a.step_ms else span * 1e3
    print(f"analytic floor {a.floor_mb:.0f} MB / {a.bw/1e12:.2f} TB/s = {fl:.3f} ms; wall {wall:.3f} ms; light {fl/wall:.2f}")
if a.json:
    json.dump(dict(title=a.title, kernels=out, kernel_ms=tot, span_ms=span*1e3, idle_ms=gap*1e3, launches=len(ev), floor_mb=a.floor_mb, step_ms=a.step_ms), open(a.json, "w"), indent=1)
