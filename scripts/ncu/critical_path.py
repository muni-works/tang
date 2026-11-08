#!/usr/bin/env python3
"""Per-kernel fill/drain floor and the decode step's critical path, against
the traces of PR #186 (and #189's 3090 occupancy data).

per kernel (mirrors tang_compile::tensor::io_lp::KernelFloor):
  t = launch + max(bytes/BW + lat,          stream: one DRAM round trip to fill, then peak BW
                   flops/peak,
                   ceil(waves) · lat)        each wave of CTAs needs >= one dependent round trip
  waves = grid / (SMs · CTAs per SM)         CTAs per SM from threads and registers
step: the kernels of one stream form a chain, so the floor is the sum of t_i
along it; kernels with no dependence may overlap (lever a).
Nothing here is fitted to the measured times: lat and BW are datasheet /
microbenchmark values, launch is the smallest kernel in the trace (an empty
graph node's cost), bytes are ncu counts (kernels) or #186's analytic floor
(step, spread over a group's launches by thread count).

  critical_path.py NCU_DIR      NCU_DIR = crates/tang-llm/docs/bench/ncu from #186
"""
import csv, json, re, sys
M = {
    "3090": dict(sms=82, thr=1536, regs=65536, bw=936e9, lat=470e-9, launch=2.0e-6, l2bw=2.5e12, l2lat=200e-9),
    "h200": dict(sms=132, thr=2048, regs=65536, bw=4.8e12, lat=600e-9, launch=1.0e-6, l2bw=10e12, l2lat=260e-9),
}
def waves(m, grid, block, regs):
    per = max(1, min(m["thr"] // block, m["regs"] // max(1, regs * block), 32))
    return grid / (m["sms"] * per)
def kfloor(m, B, grid, block, regs, l2=False):
    bw, lat = (m["l2bw"], m["l2lat"]) if l2 else (m["bw"], m["lat"])
    terms = {"stream": B / bw + lat if B > 0 else 0.0,
             "waves": max(1, -(-waves(m, grid, block, regs) // 1)) * lat if B > 0 else 0.0}
    k = max(terms, key=terms.get)
    return m["launch"] + terms[k], ("fill" if terms[k] - (B / bw if k == "stream" else 0) > (B / bw) else k)
def name(r): return re.sub(r"\(.*$", "", r["Name"].strip('"')).split("::")[-1]
d = sys.argv[1]
print("## per kernel\n")
# SM issue rate (SMs × boost clock): if a kernel's 3090 time is instruction-issue
# limited, its H200 time is about 3090 µs × this ratio. An observation, not a floor.
SMRATE = (82 * 1.695e9) / (132 * 1.98e9)
print("| kernel | MB | grid×block | regs | GPU | waves | measured µs | meas. light | **pred µs** | pred light | pred/meas | binding | 3090 µs × SM-rate |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
BIG = r"ex_gate_up|ex_down|gemv_q8_0|sh_gate_up|sh_down|sh_gate|sh_up"
for f in ["moe_t1", "moe_t8", "moe_t48", "ops_t1"]:
    j3 = {k["kernel"]: k for k in json.load(open(f"{d}/3090/{f}.json"))["kernels"]}
    jh = {k["kernel"]: k for k in json.load(open(f"{d}/h200/{f}.json"))["kernels"]}
    regs = {name(r): int(r["Reg/Trd"]) for r in csv.DictReader(open(f"{d}/h200-raw/{f}.csv")) if r.get("Name")}
    for k, kh in jh.items():
        if not re.search(BIG, k) or k not in j3: continue
        B = j3[k]["mb"] * 1e6; g, bl, rg = int(kh["grid"]), int(kh["block"]), regs.get(k, 32)
        for gpu, kk in (("3090", j3[k]), ("h200", kh)):
            m = M[gpu]; us = kk["us"]; t, why = kfloor(m, B, g, bl, rg); tm = B / m["bw"]
            print(f"| {f} `{k}` | {B/1e6:.1f} | {g}×{bl} | {rg} | {gpu} | {waves(m,g,bl,rg):.2f} | {us:.1f} | {tm*1e6/us:.2f} | **{t*1e6:.1f}** | {tm/t:.2f} | {t*1e6/us:.2f} | {why} | {(f"{j3[k]['us']*SMRATE:.1f}" if gpu == "h200" else "")} |")
p = dict(expert_mb=1.909, rest_mb=2359, shared_mb=3.34, state_mb=2.097, ctx=1380, kvtok=2048, e=512, k=10)
def group(nm, us):
    if us < 5: return "small"
    if re.search(r"moe_(gu|down|plan)|ex_|sh_|silu|route|sort_|TensorId|weights", nm): return "moe"
    if re.search(r"gemv|mma_q8x|argmax", nm): return "dense"
    if "gdn" in nm: return "gdn"
    if "attn" in nm: return "attn"
    return "small"
def thr(r): return int(r["GrdX"]) * int(r["GrdY"]) * int(r["GrdZ"]) * int(r["BlkX"]) * int(r["BlkY"]) * int(r["BlkZ"])
def grid(r): return int(r["GrdX"]) * int(r["GrdY"]) * int(r["GrdZ"])
def block(r): return int(r["BlkX"]) * int(r["BlkY"]) * int(r["BlkZ"])
print("\n## decode step, H200: serial chain (one stream), and two levers\n")
print("| config | wall ms | Σ kernels measured | bytes/BW floor | **chain floor** | pred light | meas. light | fill share of floor | (a) overlap shared expert | (a') pair sub-wave neighbours | (b) next kernel's weights in L2 | (a')+(b) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
m = M["h200"]
for cfg in ["hand_b1", "hand_b8", "hand_b48", "der_b1", "der_b8", "der_b48"]:
    n = int(cfg.split("_b")[1])
    wall = float(re.search(r"([0-9.]+) ms", open(f"{d}/h200-raw/step_{cfg}.wall").read()).group(1))
    u = p["e"] * (1 - (1 - p["k"] / p["e"]) ** n)
    fmb = dict(moe=48 * (u * p["expert_mb"] + p["shared_mb"]), dense=p["rest_mb"] - 48 * p["shared_mb"],
               gdn=36 * n * 2 * p["state_mb"], attn=12 * n * p["ctx"] * p["kvtok"] / 1e6)
    rows = sorted((r for r in csv.DictReader(open(f"{d}/h200-raw/step_{cfg}.csv")) if r.get("Name")), key=lambda r: float(r["Start (ns)"]))
    per = {}
    for r in rows: per.setdefault(name(r), []).append(float(r["Duration (ns)"]) / 1e3)
    grp = {nm: group(nm, sum(v) / len(v)) for nm, v in per.items()}
    cnt = {}
    for r in rows: g = grp[name(r)]; cnt[g] = cnt.get(g, 0) + thr(r)
    meas = sum(float(r["Duration (ns)"]) for r in rows) / 1e6
    ks = []
    for r in rows:
        g = grp[name(r)]; B = fmb[g] * 1e6 * thr(r) / cnt[g] if g in fmb else 0.0
        t, _ = kfloor(m, B, grid(r), block(r), int(r["Reg/Trd"]))
        tl2, _ = kfloor(m, B, grid(r), block(r), int(r["Reg/Trd"]), l2=True)
        w = waves(m, grid(r), block(r), int(r["Reg/Trd"]))
        ks.append(dict(nm=name(r), t=t, tl2=tl2, B=B, w=w))
    total_B = sum(fmb.values()) * 1e6; floor_bw = total_B / m["bw"]
    chain = sum(k["t"] for k in ks); fill = sum(k["t"] - k["B"] / m["bw"] for k in ks)
    # (a) the shared expert's kernels run beside the routed chain: drop them from the chain,
    # but each run of them must fit inside the routed kernels it overlaps (max per layer)
    sh = sum(k["t"] for k in ks if re.search(r"sh_|shared", k["nm"]))
    a = chain - sh
    # (a') any two adjacent kernels that together fill at most one wave run concurrently:
    # the pair costs max, not sum (greedy, an upper bound on what reordering independent work can buy)
    ap, i = 0.0, 0
    while i < len(ks):
        if i + 1 < len(ks) and ks[i]["w"] + ks[i + 1]["w"] <= 1.0:
            ap += max(ks[i]["t"], ks[i + 1]["t"]); i += 2
        else:
            ap += ks[i]["t"]; i += 1
    # (b) each kernel's weights were prefetched into L2 during its predecessor: its own fill
    # and stream come from L2; DRAM still has to deliver every byte, so the step is at least bytes/BW
    b = max(sum(k["tl2"] for k in ks), floor_bw)
    ab, i = 0.0, 0
    while i < len(ks):
        if i + 1 < len(ks) and ks[i]["w"] + ks[i + 1]["w"] <= 1.0:
            ab += max(ks[i]["tl2"], ks[i + 1]["tl2"]); i += 2
        else:
            ab += ks[i]["tl2"]; i += 1
    ab = max(ab, floor_bw)
    f = lambda x: f"{x*1e3:.2f} ms (−{(1-x/chain)*100:.0f}%)"
    print(f"| {cfg} | {wall:.2f} | {meas:.2f} | {floor_bw*1e3:.2f} | **{chain*1e3:.2f}** | {floor_bw/chain:.2f} | {floor_bw*1e3/wall:.2f} | {fill/chain*100:.0f}% | {f(a)} | {f(ap)} | {f(b)} | {f(ab)} |")
