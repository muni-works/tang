#!/usr/bin/env python3
"""Predict per-kernel and per-step time from the composite floor and compare
with the measured traces of PR #186 (fraction of light).

floor(kernel) = max(bytes / BW,                      memory
                    bytes · latency / bytes_in_flight, bytes in flight (Little's law)
                    launch)                            launch
bytes_in_flight = resident_threads · b, resident_threads from grid, block,
registers/thread and the SM limits; b = bytes one thread keeps in flight.
Mirrors tang_compile::tensor::io_lp::{Machine, Occupancy, TimeFloor}.

  predict_floor.py NCU_DIR [b ...]      NCU_DIR = crates/tang-llm/docs/bench/ncu from #186
"""
import csv, json, re, sys, os
M = {
    "3090": dict(sms=82, thr=1536, regs=65536, bw=936e9, lat=470e-9, launch=2.0e-6),
    "h200": dict(sms=132, thr=2048, regs=65536, bw=4.8e12, lat=600e-9, launch=1.0e-6),
}
def resident(m, grid, block, regs):
    occ = min(m["thr"] // block, m["regs"] // max(1, regs * block), 32)
    return min(grid, m["sms"] * max(occ, 1)) * block
def floor(m, B, grid, block, regs, b):
    tm = B / m["bw"]; tl = B * m["lat"] / (resident(m, grid, block, regs) * b)
    t = max(tm, tl, m["launch"])
    return t, ("memory" if t == tm else "in-flight" if t == tl else "launch")
def name(r): return re.sub(r"\(.*$", "", r["Name"].strip('"')).split("::")[-1]
d = sys.argv[1]; bs = [float(x) for x in sys.argv[2:]] or [64.0, 32.0, 16.0]
BIG = r"ex_gate_up|ex_down|gemv_q8_0|sh_gate_up|sh_down"
print("## per kernel (bytes: 3090 ncu; regs/thread: H200 nsys)\n")
print("| kernel | MB | grid×block | regs | GPU | measured µs | meas. light | " + " | ".join(f"pred µs / light (b={b:g})" for b in bs) + " |")
print("|---|---|---|---|---|---|---|" + "---|" * len(bs))
for f in ["moe_t1", "moe_t8", "moe_t48", "ops_t1"]:
    j3 = {k["kernel"]: k for k in json.load(open(f"{d}/3090/{f}.json"))["kernels"]}
    jh = {k["kernel"]: k for k in json.load(open(f"{d}/h200/{f}.json"))["kernels"]}
    regs = {}
    for r in csv.DictReader(open(f"{d}/h200-raw/{f}.csv")):
        if r.get("Name"): regs[name(r)] = int(r["Reg/Trd"])
    for k, kh in jh.items():
        if not re.search(BIG, k) or k not in j3: continue
        B = j3[k]["mb"] * 1e6; g, bl, rg = int(kh["grid"]), int(kh["block"]), regs.get(k, 32)
        for gpu, kk in (("3090", j3[k]), ("h200", kh)):
            us = kk["us"]; m = M[gpu]; tm = B / m["bw"]
            cells = []
            for b in bs:
                t, why = floor(m, B, g, bl, rg, b)
                flag = "" if t <= us * 1e-6 * 1.02 else " ✗"
                cells.append(f"{t*1e6:.1f} / {tm/t:.2f} {why}{flag}")
            print(f"| {f} `{k}` | {B/1e6:.1f} | {g}×{bl} | {rg} | {gpu} | {us:.1f} | {tm*1e6/us:.2f} | " + " | ".join(cells) + " |")
# whole decode step on the H200: group floors from step_groups.py, spread over the group's launches
p = dict(expert_mb=1.909, rest_mb=2359, shared_mb=3.34, state_mb=2.097, ctx=1380, kvtok=2048, e=512, k=10)
def group(nm, us):
    if us < 5: return "small"
    if re.search(r"moe_(gu|down|plan)|ex_|sh_|silu|route|sort_|TensorId|weights", nm): return "moe"
    if re.search(r"gemv|mma_q8x|argmax", nm): return "dense"
    if "gdn" in nm: return "gdn"
    if "attn" in nm: return "attn"
    return "small"
print("\n## decode step, H200 (measured: #186 nsys; ✗ = predicted floor above measured)\n")
print("| config | wall ms | floor ms (bytes/BW) | measured light | " + " | ".join(f"pred ms / light (b={b:g})" for b in bs) + " | binding launches (b=16) |")
print("|---|---|---|---|" + "---|" * len(bs) + "---|")
m = M["h200"]
for cfg in ["hand_b1", "hand_b8", "hand_b48", "der_b1", "der_b8", "der_b48"]:
    n = int(cfg.split("_b")[1]); wall = float(re.search(r"([0-9.]+) ms", open(f"{d}/h200-raw/step_{cfg}.wall").read()).group(1))
    u = p["e"] * (1 - (1 - p["k"] / p["e"]) ** n)
    fmb = dict(moe=48 * (u * p["expert_mb"] + p["shared_mb"]), dense=p["rest_mb"] - 48 * p["shared_mb"],
               gdn=36 * n * 2 * p["state_mb"], attn=12 * n * p["ctx"] * p["kvtok"] / 1e6)
    rows = [r for r in csv.DictReader(open(f"{d}/h200-raw/step_{cfg}.csv")) if r.get("Name")]
    per = {}
    for r in rows: per.setdefault(name(r), []).append(r)
    grp = {nm: group(nm, sum(float(r["Duration (ns)"]) for r in rs) / len(rs) / 1e3) for nm, rs in per.items()}
    # a group's floor bytes are spread over its launches in proportion to their threads
    # (equal bytes per thread): the trace has no per-kernel bytes for the step
    def thr(r): return int(r["GrdX"]) * int(r["GrdY"]) * int(r["GrdZ"]) * int(r["BlkX"]) * int(r["BlkY"]) * int(r["BlkZ"])
    cnt = {}
    for nm, rs in per.items(): cnt[grp[nm]] = cnt.get(grp[nm], 0) + sum(thr(r) for r in rs)
    floor_ms = sum(fmb.values()) * 1e6 / m["bw"] * 1e3
    cells, why_ct = [], {}
    for b in bs:
        tot = 0
        for nm, rs in per.items():
            g = grp[nm]
            for r in rs:
                B = fmb[g] * 1e6 * thr(r) / cnt[g] if g in fmb else 0.0
                t, why = floor(m, B, int(r["GrdX"]) * int(r["GrdY"]) * int(r["GrdZ"]),
                               int(r["BlkX"]) * int(r["BlkY"]) * int(r["BlkZ"]), int(r["Reg/Trd"]), b)
                tot += t
                if b == 16: why_ct[why] = why_ct.get(why, 0) + 1
        flag = " ✗" if tot * 1e3 > wall * 1.02 else ""
        cells.append(f"{tot*1e3:.2f} / {floor_ms/(tot*1e3):.2f}{flag}")
    print(f"| {cfg} | {wall:.2f} | {floor_ms:.2f} | {floor_ms/wall:.2f} | " + " | ".join(cells) + " | " + ", ".join(f"{k} {v}" for k, v in sorted(why_ct.items())) + " |")
