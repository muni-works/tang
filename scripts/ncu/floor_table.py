#!/usr/bin/env python3
"""The per-kernel floor table in docs/duality.md, from the committed traces.

  python3 scripts/ncu/floor_table.py [crates/tang-llm/docs/bench/ncu]

floor = L + max(lam + B/BW, ceil(G/S)*lam)     (mirrors StepKernel::floor)
  L    empty-kernel cost: 3090 0.77 us (#160, independent of these traces);
       H200 0.48 us (empty kernel in a CUDA graph, #196)
  lam  3090 470 ns (assumed); H200 340 ns (pointer chase over 4 GB, #196),
       145 ns (measured L2 hop, #196) for a kernel that fits the H200's 50 MB L2
  S    resident workgroups: SMs * min(threads/block, regs/(regs_per_thread*block), hw limit)
Bytes are 3090 ncu counts; the H200 column reuses them (no counters on the pod).
Kernels that fit the H200's L2 are marked: B/BW is then not a proven floor.
"L1?" adds the unvalidated wavefront hypothesis (Q8_0 GEMV only).
`dop00_gemv_q8_0` in ops_t1 is the mean of three shapes, so it is split by shape.
"""
import csv, json, math, re, sys
d = sys.argv[1] if len(sys.argv) > 1 else "crates/tang-llm/docs/bench/ncu"
M = {"3090": dict(sms=82, thr=1536, hw=32, bw=936e9, lam=470e-9, L=0.77e-6, clk=1.695e9),
     "h200": dict(sms=132, thr=2048, hw=256, bw=4.8e12, lam=340e-9, L=0.48e-6, clk=1.98e9)}
L2 = 50e6
def S(m, bl, rg): return m["sms"] * max(1, min(m["thr"] // bl, 65536 // (max(rg, 1) * bl), m["hw"]))
def floor(gpu, B, G, bl, rg):
    m = M[gpu]; lam = 145e-9 if gpu == "h200" and B <= L2 else m["lam"]
    a = lam + B / m["bw"]; w = math.ceil(G / S(m, bl, rg)) * lam
    t = m["L"] + max(a, w)
    why = "waves" if w > a else ("fill" if lam > B / m["bw"] else "stream")
    return t, why
def name(r): return re.sub(r"\(.*$", "", r["Name"].strip('"')).split("::")[-1]
def regs(f):
    return {name(r): int(r["Reg/Trd"]) for r in csv.DictReader(open(f"{d}/h200-raw/{f}.csv")) if r.get("Name")}
rows = []
for f in ["moe_t1", "moe_t8", "moe_t48", "ops_t1"]:
    j3 = {k["kernel"]: k for k in json.load(open(f"{d}/3090/{f}.json"))["kernels"]}
    jh = {k["kernel"]: k for k in json.load(open(f"{d}/h200/{f}.json"))["kernels"]}
    rg = regs(f)
    for k, h in jh.items():
        if k not in j3 or k == "dop00_gemv_q8_0" or j3[k]["mb"] < 0.5: continue
        B = j3[k]["mb"] * 1e6
        rows.append((f"{f} {k}", B, int(h["grid"]), int(h["block"]), rg.get(k, 32), j3[k]["us"], h["us"], None))
Q = lambda r, k: r * k / 32 * 34
# 3090: #189 gemv_bif (includes a stream launch and event, so loose); H200: #186 nsys, launches 2-4
rows += [("ops_t1 q8_0 4096x2048", Q(4096, 2048), 256, 256, 36, Q(4096, 2048) / 495e9 * 1e6, 7.4, "q8"),
         ("ops_t1 q8_0 2048x4096", Q(2048, 4096), 256, 256, 34, None, 9.9, "q8"),
         ("ops_t1 q8_0 151936x2048", Q(151936, 2048), 9496, 256, 36, Q(151936, 2048) / 864e9 * 1e6, 172.2, "q8")]
print("| kernel | MB | G×block | H200 L2? | 3090 meas µs | floor µs (term) | floor/meas | H200 meas µs | floor µs (term) | floor/meas | L1? µs | L1?/meas |")
print("|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
out = {"3090": [], "h200": []}
for n, B, G, bl, rg, u3, uh, q in rows:
    t3, w3 = floor("3090", B, G, bl, rg); th, wh = floor("h200", B, G, bl, rg)
    if u3: out["3090"].append((t3 * 1e6 / u3, n))
    out["h200"].append((th * 1e6 / uh, n))
    c3 = f"{u3:.1f} | {t3*1e6:.1f} ({w3}) | {t3*1e6/u3:.2f}" if u3 else f"– | {t3*1e6:.1f} ({w3}) | –"
    l1 = ""
    if q:
        m = M["h200"]; tl = m["L"] + 340e-9 + B / (8 * m["sms"] * m["clk"]); tl = max(tl, th)
        l1 = f"{tl*1e6:.1f} | {tl*1e6/uh:.2f}"
    else: l1 = " | "
    print(f"| {n} | {B/1e6:.1f} | {G}×{bl} | {'yes' if B<=L2 else 'no'} | {c3} | {uh:.1f} | {th*1e6:.1f} ({wh}) | {th*1e6/uh:.2f} | {l1} |")
print()
for g, v in out.items():
    v.sort(); r = [x for x, _ in v]
    print(f"{g}: {len(v)} kernels, floor/meas min {r[0]:.2f} median {r[len(r)//2]:.2f} max {r[-1]:.2f}; over 1: {[n for x,n in v if x>1]}")
