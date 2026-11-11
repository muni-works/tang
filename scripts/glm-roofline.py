#!/usr/bin/env python3
"""Decode roofline for GLM-5.3-Flash, the row to match ds-roofline.py's.

Bytes a decode forward must read, per token and at batch B, at several
weight precisions; tok/s = bandwidth / bytes. The UD-Q4_K_XL row uses the
GGUF's own tensor sizes (unsloth/GLM-5.3-Flash-GGUF, measured from the
shard headers by scripts/glm-fetch-layers.py's parser); the others scale
the parameter counts. Everything here is arithmetic (assumed hardware
peaks), not a measurement.

    glm-roofline.py
"""

# geometry (GGUF metadata, glm5-next)
N, L, HC = 4096, 45, 4
DSA = [l for l in range(L) if l % 4 == 3]          # 11 DSA layers (3, 7, ..., 43)
KDA = [l for l in range(L) if l not in DSA]          # 34 KDA layers
DENSE = 3                                            # leading dense FFN layers (all KDA)
HEADS, DQK, DV, QL, KL = 64, 256, 256, 1536, 512
IH, ID, TOPK, KPOOL = 32, 128, 2048, 4
KHD = 128                                            # KDA head dim (64 heads)
E, ACT, FE, FF = 288, 8, 2048, 12288
VOCAB = 154880

# parameter counts by role, per layer
kda_mixer = 4 * N * HEADS * KHD + 2 * (N * KHD + KHD * HEADS * KHD) + N * HEADS  # q k v o, f/g low-rank, beta
dsa_mixer = N * QL + QL * HEADS * DQK + N * KL + HEADS * KL * (DQK + DV) + HEADS * DV * N
indexer = N * ID + QL * IH * ID + N * IH + N * ID
dense_ffn = 3 * N * FF
expert = 3 * N * FE
shared = 3 * N * FE
router = E * N
mhc = 2 * (2 + HC) * HC * HC * N  # attn + ffn mixes, f32
head = VOCAB * N

# UD-Q4_K_XL bytes measured per role (MB): mixers / dense FFN / shared / head Q8_0,
# experts Q4_K gate+up, Q5_K down (layers 11 and 44 heavier), mHC and router F32
Q = dict(kda_mixer=146.65e6, dsa_mixer=124.79e6, indexer=8.33e6, dense_ffn=160.45e6,
         shared=26.74e6, router=4.72e6, mhc=0.84e6, head=674.04e6,
         expert=4378.85e6 / E, expert_heavy={11: 5303.7e6 / E, 12: 4699.72e6 / E, 44: 4699.72e6 / E})


def per_token_params():
    a = 0
    for l in range(L):
        a += (kda_mixer if l in KDA else dsa_mixer + indexer) + mhc
        a += dense_ffn if l < DENSE else router + shared + ACT * expert
    return a + head


def total_params():
    t = 0
    for l in range(L):
        t += (kda_mixer if l in KDA else dsa_mixer + indexer) + mhc
        t += dense_ffn if l < DENSE else router + shared + E * expert
    return t + 2 * head


def distinct(B):
    return E * (1 - (1 - ACT / E) ** B)


PREC = {
    "bf16": dict(w=2.0, x=2.0),
    "UD-Q4_K_XL (the GGUF: Q8_0 dense, ~4.6-bit experts)": None,
    "fp8 all": dict(w=1 + 1 / 1024, x=1 + 1 / 1024),
    "4-bit all (Q4_K)": dict(w=4.5 / 8, x=4.5 / 8),
}


def dense_bytes(p):
    """Weights every token reads once whatever B is (mixers, dense FFN, shared, router, mHC, head)."""
    if p is None:
        b = 0
        for l in range(L):
            b += (Q["kda_mixer"] if l in KDA else Q["dsa_mixer"] + Q["indexer"]) + Q["mhc"]
            b += Q["dense_ffn"] if l < DENSE else Q["router"] + Q["shared"]
        return b + Q["head"]
    b = 0
    for l in range(L):
        b += ((kda_mixer if l in KDA else dsa_mixer + indexer)) * p["w"] + mhc * 4
        b += dense_ffn * p["w"] if l < DENSE else router * 4 + shared * p["w"]
    return b + head * p["w"]


def expert_bytes(p, B):
    n = distinct(B)
    if p is None:
        return sum(n * Q["expert_heavy"].get(l, Q["expert"]) for l in range(DENSE, L))
    return (L - DENSE) * n * expert * p["x"]


def state_bytes(ctx, kv=2.0):
    """Per sequence per decode token: every KDA layer reads and writes its f32 state
    (64 heads x 128 x 128); every DSA layer reads its latent rows (512, top-2048 once
    ctx > 2048) and its indexer's pooled keys (128 per 4 tokens), `kv` bytes a value."""
    kda = len(KDA) * HEADS * KHD * KHD * 4 * 2
    rows = min(ctx, TOPK)
    dsa = len(DSA) * (rows * KL + (ctx // KPOOL) * ID) * kv
    return kda + dsa


GPUS = {"H100 SXM 80GB": 3.35e12, "H200 141GB": 4.8e12, "RTX PRO 6000 96GB": 1.79e12, "B200 192GB": 8.0e12}
# the 200 GB GGUF needs >= 288 GB of HBM in practice: no single GPU above holds it (the
# per-GPU columns are bandwidth rooflines, as in ds-roofline.py); a B300 (288 GB) does


def state_step(ctx, w):
    """A w-token window: the KDA states are read and written once, the DSA rows w times."""
    kda = len(KDA) * HEADS * KHD * KHD * 4 * 2
    return kda + w * (state_bytes(ctx) - kda)


def step_bytes(p, B, ctx):
    return dense_bytes(p) + expert_bytes(p, B) + B * state_bytes(ctx)


def main():
    print(f"active params/token: {per_token_params() / 1e9:.2f} B; total (no MTP layer): {total_params() / 1e9:.1f} B")
    print(f"  per layer: KDA mixer {kda_mixer / 1e6:.1f} M, DSA mixer {dsa_mixer / 1e6:.1f} M + indexer {indexer / 1e6:.1f} M, "
          f"expert {expert / 1e6:.1f} M x {ACT}, shared {shared / 1e6:.1f} M, router {router / 1e6:.2f} M")
    for ctx in (4096, 32768):
        print(f"KDA state + DSA latent/indexer bytes per token per sequence at {ctx}: {state_bytes(ctx) / 1e6:.1f} MB")
    print()
    print("| precision | weights/token (B=1) | resident weights | H100 | H200 | RTX PRO 6000 | B200 |")
    print("|---|---|---|---|---|---|---|")
    for name, p in PREC.items():
        b1 = step_bytes(p, 1, 4096)
        if p is None:
            res = 199.7e9
        else:
            res = dense_bytes(p) + (L - DENSE) * E * expert * p["x"] + head * p["w"]
        row = [f"{b1 / 1e9:.2f} GB", f"{res / 1e9:.0f} GB"] + [f"{bw / b1:.0f}" for bw in GPUS.values()]
        print(f"| {name} | " + " | ".join(row) + " |")
    print()
    q = PREC["UD-Q4_K_XL (the GGUF: Q8_0 dense, ~4.6-bit experts)"]
    print("Batched (UD-Q4_K_XL, aggregate bandwidth, perfect split, 4k / 32k ctx): forwards x B tok/s")
    print("| GPUs | B=1 | 8 | 32 | 64 | 128 | 256 |")
    print("|---|---|---|---|---|---|---|")
    for name, n, bw in [("1x B300 288GB", 1, 8e12), ("2x H200", 2, 4.8e12), ("4x H100", 4, 3.35e12),
                        ("2x B200", 2, 8e12), ("3x RTX PRO 6000", 3, 1.79e12), ("4x RTX PRO 6000", 4, 1.79e12)]:
        cells = []
        for B in (1, 8, 32, 64, 128, 256):
            a = B * n * bw / step_bytes(q, B, 4096)
            c = B * n * bw / step_bytes(q, B, 32768)
            cells.append(f"{a:,.0f} / {c:,.0f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    print("Batch 1 with speculation (UD-Q4_K_XL, 4k ctx).")
    print("A forward of a w-token window reads the dense weights once, every distinct routed expert the")
    print("window picks, the KDA states once and w tokens of DSA rows.")
    specs = [("plain", 1, 1.0), ("MTP K=3 (w 4, 1.91 tok/step, measured on code)", 4, 1.91),
             ("MTP K=3 verify-only bound (2.98 tok/step)", 4, 2.98)]
    print("| GPU | " + " | ".join(s[0] for s in specs) + " |")
    print("|---|" + "---|" * len(specs))
    for name, bw in [("1x B300 (hand engine measured: ~100 plain, 129.6 MTP K=1)", 8e12),
                     ("2x H200 (EP + TP dense, 12 us AR x 2 per layer)", 4.8e12)]:
        cells = []
        for _, w, acc in specs:
            t = (dense_bytes(q) + expert_bytes(q, w) + state_step(4096, w)) / bw
            if "2x" in name:
                t = t / 2 + 2 * L * 12e-6
            cells.append(f"{acc / t:,.0f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
