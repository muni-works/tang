#!/usr/bin/env python3
"""Aggregate ncu_kernels.sh CSV per kernel: launches, time, DRAM bytes, the
traffic floor (bytes / peak HBM bandwidth) and the fraction of light
(floor / measured), with a bound classification.

  table.py CSV --bw 4.8e12 [--title T] [--step-ms 28.45 --steps 1] [--json out.json]

Classes (per kernel, from its mean launch):
  launch   mean duration < --launch-us (default 5 us): fixed launch/ramp/tail cost dominates
  traffic  DRAM throughput >= 50% of peak (or floor/measured >= 0.5)
  compute  SM throughput >= 60% of peak
  latency  none of the above: the kernel neither fills DRAM nor the SMs
           (too few blocks, dependent loads, serialized reductions)
With --step-ms, the gap between the step's wall time and the sum of kernel
durations is reported as launch/idle overhead (inter-kernel bubbles).
"""
import argparse, csv, json, re, sys
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--bw", type=float, required=True, help="peak HBM bytes/s")
ap.add_argument("--title", default="")
ap.add_argument("--step-ms", type=float)
ap.add_argument("--steps", type=float, default=1.0, help="steps the profiled launches cover")
ap.add_argument("--launch-us", type=float, default=5.0)
ap.add_argument("--json")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
launches = OrderedDict()  # ID -> {name, metrics}
for r in rows:
    lid = r["ID"]
    d = launches.setdefault(lid, {"name": r["Kernel Name"], "m": {}})
    try:
        d["m"][r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
    except ValueError:
        pass

def short(n):
    n = re.sub(r"\(.*$", "", n)          # drop the signature
    n = re.sub(r"^void\s+", "", n)
    n = re.sub(r"<.*>$", "", n)
    return n.split("::")[-1]

K = OrderedDict()
for d in launches.values():
    m = d["m"]
    key = short(d["name"])
    if key.startswith("main"):  # the IR runtime's kernels share an entry name: split by shape
        key = f"{key}@{int(m.get('launch__grid_size', 0))}x{int(m.get('launch__block_size', 0))}"
    k = K.setdefault(key, {"n": 0, "t": 0.0, "rd": 0.0, "wr": 0.0, "sm": 0.0, "dr": 0.0, "grid": m.get("launch__grid_size", 0), "block": m.get("launch__block_size", 0)})
    k["n"] += 1
    k["t"] += m.get("gpu__time_duration.sum", 0) * 1e-9
    k["rd"] += m.get("dram__bytes_read.sum", 0)
    k["wr"] += m.get("dram__bytes_write.sum", 0)
    k["sm"] += m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    k["dr"] += m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    k["occ"] = k.get("occ", 0) + m.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
    k["regs"] = m.get("launch__registers_per_thread", 0)
    k["waves"] = m.get("launch__waves_per_multiprocessor", 0)
    k["inst"] = k.get("inst", 0) + m.get("sm__inst_executed.sum", 0)
    k["issue"] = k.get("issue", 0) + m.get("sm__instruction_throughput.avg.pct_of_peak_sustained_active", 0)
    k["lsb"] = k.get("lsb", 0) + m.get("smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio", 0)

out = []
for name, k in K.items():
    n = k["n"]
    t = k["t"] / n
    by = (k["rd"] + k["wr"]) / n
    floor = by / a.bw
    light = floor / t if t > 0 else 0
    sm, dr = k["sm"] / n, k["dr"] / n
    if t * 1e6 < a.launch_us:
        cls = "launch"
    elif dr >= 50 or light >= 0.5:
        cls = "traffic"
    elif sm >= 60:
        cls = "compute"
    else:
        cls = "latency"
    out.append(dict(kernel=name, launches=n, us=t * 1e6, total_ms=k["t"] * 1e3, mb=by / 1e6,
                    floor_us=floor * 1e6, light=light, sm_pct=sm, dram_pct=dr,
                    grid=int(k["grid"]), block=int(k["block"]), cls=cls,
                    occ_pct=k.get("occ", 0) / n, regs=k.get("regs", 0), waves=k.get("waves", 0),
                    long_sb=k.get("lsb", 0) / n, inst=k.get("inst", 0) / n, issue_pct=k.get("issue", 0) / n))
out.sort(key=lambda r: -r["total_ms"])
tot = sum(r["total_ms"] for r in out)
tot_floor = sum(r["floor_us"] * r["launches"] for r in out) / 1e3
if a.title:
    print(f"#### {a.title}\n")
print("| kernel | launches | µs each | total ms | share | MB each | floor µs | light | SM % | DRAM % | occ % | regs | waves | long-sb cyc/warp | warp inst | inst/byte | issue % | grid×block | bound |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in out:
    print(f"| `{r['kernel']}` | {r['launches']} | {r['us']:.1f} | {r['total_ms']:.3f} | {100*r['total_ms']/tot:.1f}% | {r['mb']:.2f} | {r['floor_us']:.1f} | {r['light']:.2f} | {r['sm_pct']:.0f} | {r['dram_pct']:.0f} | {r['occ_pct']:.0f} | {r['regs']:.0f} | {r['waves']:.2f} | {r['long_sb']:.0f} | {r['inst']:.0f} | {r['inst'] / max(r['mb'] * 1e6, 1):.3f} | {r['issue_pct']:.0f} | {r['grid']}×{r['block']} | {r['cls']} |")
by_cls = {}
for r in out:
    by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + r["total_ms"]
print(f"\nkernel time {tot:.3f} ms, traffic floor {tot_floor:.3f} ms (light {tot_floor/tot:.2f}); by class: " +
      ", ".join(f"{c} {v:.3f} ms ({100*v/tot:.0f}%)" for c, v in sorted(by_cls.items(), key=lambda x: -x[1])))
if a.step_ms:
    wall = a.step_ms * a.steps
    print(f"wall {wall:.3f} ms over {a.steps:g} step(s): kernels {tot:.3f} ms, launch/idle gap {wall - tot:.3f} ms ({100*(wall-tot)/wall:.0f}%); floor/wall {tot_floor/wall:.2f}")
if a.json:
    json.dump(dict(title=a.title, bw=a.bw, kernels=out, kernel_ms=tot, floor_ms=tot_floor, step_ms=a.step_ms, steps=a.steps), open(a.json, "w"), indent=1)
