"""Markdown tables from attn_variants_bench.py's CSV: ms (TFLOPS) per system."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
T = {(r["variant"], r["d"], r["n"], r["pass"], r["system"]): r for r in rows}
SYS = [("tang", "tang (fp32, TF32 mma)"), ("flex_fp32", "Flex fp32/TF32"), ("flex_bf16", "Flex bf16"), ("fa2_bf16", "FA2 bf16")]
VAR = ["causal", "sliding128", "alibi", "softcap", "prefix64", "bidirectional"]
for d in ("64", "128"):
    for p, pn in (("fwd", "forward"), ("fwdbwd", "forward + backward")):
        print(f"\n**d = {d}, {pn}** — ms (TFLOPS); best per row in bold\n")
        print("| variant | n | " + " | ".join(s[1] for s in SYS) + " |")
        print("|---|---:|" + "---:|" * len(SYS))
        for v in VAR:
            for n in ("1024", "2048", "4096", "8192"):
                cells, best = [], min((float(T[k]["ms"]) for k in [(v, d, n, p, s) for s, _ in SYS] if k in T and T[k]["ms"]), default=None)
                for s, _ in SYS:
                    r = T.get((v, d, n, p, s))
                    if not r or not r["ms"]:
                        note = (r or {}).get("note", "")
                        cells.append("n/a" if "unsupported" in note else "fails" if note else "–")
                        continue
                    c = f"{float(r['ms']):.3g} ({float(r['tflops']):.1f})"
                    cells.append(f"**{c}**" if float(r["ms"]) == best else c)
                print(f"| {v} | {n} | " + " | ".join(cells) + " |")
