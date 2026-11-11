#!/usr/bin/env python3
"""Decode roofline for DeepSeek-V4.1-Flash (bull), from config.json shapes.

Bytes a decode forward must read, per token and at batch B, at several
weight precisions; tok/s = bandwidth / bytes. Everything here is arithmetic
on the config (assumed hardware peaks), not a measurement.

    ds-roofline.py [config.json]
"""
import json
import sys

cfg = {
    "dim": 5120, "layers": 40, "heads": 64, "head_dim": 512, "q_lora": 1280,
    "o_lora": 1024, "o_groups": 8, "experts": 384, "active": 6, "inter": 2304,
    "vocab": 129280, "window": 128, "index_heads": 32, "index_dim": 128,
    "index_topk": 512, "kv_sources": [2, 8, 14, 20], "index_sources": [2, 8, 14, 20, 24, 28, 32, 36],
    "ratios": [0, 0] + [2] * 18 + [1] * 20, "cand_blocks": 2048, "cand_block": 8, "hc": 4,
}
if len(sys.argv) > 1:
    t = json.load(open(sys.argv[1]))["text_config"]
    assert t["hidden_size"] == cfg["dim"] and t["n_routed_experts"] == cfg["experts"]

D, L, I = cfg["dim"], cfg["layers"], cfg["inter"]
# per-layer parameter counts by role
attn = 1280 * D + 32768 * 1280 + 512 * D + 8192 * 4096 + D * 8192  # wq_a wq_b wkv wo_a wo_b
shared = 3 * I * D
expert = 3 * I * D
gate = cfg["experts"] * D
hc_fn = 2 * (2 + cfg["hc"]) * cfg["hc"] * cfg["hc"] * D  # attn + ffn mixes, f32
compressor = 2 * 512 * D * cfg["ratios"].count(2) // 18  # wkv + wgate (ratio-2 sources)
n_kv_src, n_idx_src = len(cfg["kv_sources"]), len(cfg["index_sources"])
comp_total = n_kv_src * 2 * 512 * D  # compressor wkv + wgate at kv sources (bf16)
idx_total = n_idx_src * (4096 * 1280 + 128 * D + 32 * D)  # indexer wq_b (fp8), wk, weights_proj
head = cfg["vocab"] * D

active_params = L * (attn + shared + cfg["active"] * expert + gate) + comp_total + idx_total + head
total_params = L * (attn + shared + cfg["experts"] * expert + gate) + comp_total + idx_total + 2 * head


def dense_bytes(bw_dense, bw_head):
    """Weights every token reads once whatever B is (attention, shared expert, router, mHC, head)."""
    return (L * (attn + shared) + idx_total * 0.9) * bw_dense + (L * gate + comp_total + idx_total * 0.1) * 2 \
        + L * hc_fn * 4 + head * bw_head


# bytes per weight, scales included: fp8 + e8m0 per 32x32 block; fp4 e2m1 + e8m0 per 32
PREC = {
    "bf16":        dict(dense=2.0,        expert=2.0,        head=2.0),
    "native (fp8 dense, fp4 experts, bf16 head)": dict(dense=1 + 1 / 1024, expert=0.5 + 1 / 32, head=2.0),
    "fp8 all":     dict(dense=1 + 1 / 1024, expert=1 + 1 / 1024, head=1.0),
    "4-bit all":   dict(dense=0.5 + 1 / 32, expert=0.5 + 1 / 32, head=0.5 + 1 / 32),
    "~2.9-bit VQ experts, fp8 dense": dict(dense=1 + 1 / 1024, expert=2.9 / 8, head=1.0),
    "~2.9-bit VQ all": dict(dense=2.9 / 8, expert=2.9 / 8, head=2.9 / 8),
}


def kv_bytes(ctx):
    """Per sequence per decode token: window ring (all layers, fp8 K=V), the top-k compressed rows
    each compressing layer attends to (fp4 + e4m3/16), the indexer keys the 8 index layers score
    (fp4 + e8m0/32, the candidate-block prefilter capping layers after 20)."""
    win = L * min(ctx, cfg["window"]) * 512 * (1 + 1 / 64)
    comp_layers = sum(1 for r in cfg["ratios"] if r)
    rows = lambda r: ctx // r
    topk = sum(min(cfg["index_topk"], rows(r)) for r in cfg["ratios"] if r) * 512 * (0.5 + 1 / 16)
    cap = cfg["cand_blocks"] * cfg["cand_block"]
    idx = 0
    for l in cfg["index_sources"]:
        n = rows(cfg["ratios"][l])
        if l > 20:
            n = min(n, cap)
        idx += n * 128 * (0.5 + 1 / 32)
    return win + topk + idx


GPUS = {"H100 SXM 80GB": (3.35e12, 80), "H200 141GB": (4.8e12, 141), "RTX PRO 6000 96GB": (1.79e12, 96),
        "B200 192GB": (8.0e12, 192)}


def distinct(B):
    return cfg["experts"] * (1 - (1 - cfg["active"] / cfg["experts"]) ** B)


def step_bytes(p, B, ctx):
    pr = PREC[p]
    exp = L * distinct(B) * expert * pr["expert"]
    return dense_bytes(pr["dense"], pr["head"]) + exp + B * kv_bytes(ctx)


def main():
    print(f"active params/token: {active_params / 1e9:.2f} B; total (no Engram, no MTP): {total_params / 1e9:.1f} B")
    print(f"  per layer: attn {attn / 1e6:.1f} M, shared {shared / 1e6:.1f} M, expert {expert / 1e6:.1f} M x {cfg['active']}, gate {gate / 1e6:.2f} M")
    for ctx in (4096, 32768):
        print(f"KV+indexer bytes read per token per sequence at {ctx}: {kv_bytes(ctx) / 1e6:.1f} MB")
    print()
    print("| precision | weights/token (B=1) | resident weights | H100 | H200 | RTX PRO 6000 | B200 |")
    print("|---|---|---|---|---|---|---|")
    for p, pr in PREC.items():
        b1 = step_bytes(p, 1, 4096)
        res = (L * (attn + shared + gate) + comp_total + idx_total) * pr["dense"] + L * cfg["experts"] * expert * pr["expert"] + head * pr["head"] + head * 2
        row = [f"{b1 / 1e9:.2f} GB", f"{res / 1e9:.0f} GB"]
        row += [f"{bw / b1:.0f}" for bw, _ in GPUS.values()]
        print(f"| {p} | " + " | ".join(row) + " |")
    print()
    print("Batched (native precision, aggregate bandwidth, perfect split, 4k / 32k ctx): forwards x B tok/s")
    for name, n, bw in [("8x H100", 8, 3.35e12), ("8x H200", 8, 4.8e12), ("4x H200", 4, 4.8e12),
                        ("4x RTX PRO 6000", 4, 1.79e12), ("8x RTX PRO 6000", 8, 1.79e12), ("4x B200", 4, 8e12)]:
        cells = []
        for B in (1, 8, 32, 64, 128, 256):
            a = B * n * bw / step_bytes("native (fp8 dense, fp4 experts, bf16 head)", B, 4096)
            c = B * n * bw / step_bytes("native (fp8 dense, fp4 experts, bf16 head)", B, 32768)
            cells.append(f"{a:,.0f} / {c:,.0f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print()
    print("Batch 1 on real multi-GPU layouts (native precision, 4k ctx). A forward of a w-token window")
    print("reads the dense weights once and every distinct routed expert the window picks.")
    print("Collectives: 12 us per NVLink all-reduce (assumed small-message NCCL latency).")
    nat = PREC["native (fp8 dense, fp4 experts, bf16 head)"]
    dense = dense_bytes(nat["dense"], nat["head"])
    exp1 = expert * nat["expert"]
    ar = 12e-6

    def step_time(layout, n, bw, w):
        dist = distinct(w)
        # expected busiest rank under EP: ~ 1.4x the mean for small pick counts
        rank_exp = L * max(1.0, 1.4 * dist / n) * exp1 if n > 1 else L * dist * exp1
        kv = w * kv_bytes(4096)
        if layout == "1 GPU":
            return (dense + L * dist * exp1 + kv) / bw
        if layout == "EP, dense replicated":
            return (dense + rank_exp + kv) / bw + L * ar
        if layout == "EP + TP dense":
            return (dense / n + rank_exp + kv / n) / bw + 2 * L * ar
        raise ValueError(layout)

    specs = [("plain", 1, 1.0), ("prompt lookup (w 4, 1.7 tok/step, assumed)", 4, 1.7),
             ("DSpark MTP (w 7, 4.76 tok/step, measured acceptance)", 7, 4.76)]
    print("| layout | GPUs | " + " | ".join(s[0] for s in specs) + " |")
    print("|---|---|" + "---|" * len(specs))
    for layout, name, n, bw in [("EP, dense replicated", "8x H100", 8, 3.35e12),
                                ("EP + TP dense", "8x H100", 8, 3.35e12),
                                ("EP, dense replicated", "8x H200", 8, 4.8e12),
                                ("EP + TP dense", "8x H200", 8, 4.8e12),
                                ("EP + TP dense", "4x H200", 4, 4.8e12),
                                ("EP + TP dense", "4x B200", 4, 8e12),
                                ("EP + TP dense", "4x RTX PRO 6000 (PCIe, 25 us AR assumed)", 4, 1.79e12)]:
        cells = []
        for _, w, acc in specs:
            t = step_time(layout, n, bw, w)
            if "PCIe" in name:
                t += 2 * L * 13e-6
            cells.append(f"{acc / t:,.0f}")
        print(f"| {layout} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
