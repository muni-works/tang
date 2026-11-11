#!/usr/bin/env python3
"""Fetch a few layers of DeepSeek-V4.1-Flash-1017 by HTTP range requests.

Writes one safetensors file per source shard holding only the wanted tensors
(raw bytes, dtype strings as in the checkpoint: F8_E4M3, F8_E8M0, I8, BF16,
F32), plus the config. No torch, no safetensors package (it can't read
F8_E8M0).

    ds-fetch-layers.py OUT_DIR [LAYERS=0,1,2]

Skips the Engram tables (layers.*.engram.embed.*, ~100 GB each).
"""
import concurrent.futures as cf
import json
import os
import struct
import sys
import urllib.request

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"


def get(url, rng=None):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0]:
                raise IOError(f"short read {len(data)}")
            return data
        except Exception as e:  # retry transient errors
            if attempt == 5:
                raise
            print(f"retry {url} {rng}: {e}", file=sys.stderr)


def main():
    out = sys.argv[1]
    layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
    os.makedirs(out, exist_ok=True)
    for f in ["config.json", "model.safetensors.index.json"]:
        p = os.path.join(out, f)
        if not os.path.exists(p):
            open(p, "wb").write(get(REPO + f))
    wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]

    def wanted(k):
        if ".engram.embed." in k:
            return False
        if not k.startswith("layers."):
            return k in ("embed.weight", "head.weight", "norm.weight") or k.startswith("hc_head")
        return int(k.split(".")[1]) in layers

    shards = {}
    for k, s in wm.items():
        if wanted(k):
            shards.setdefault(s, []).append(k)
    total = 0
    for shard, names in sorted(shards.items()):
        dst = os.path.join(out, shard)
        if os.path.exists(dst):
            print(f"have {shard}")
            continue
        url = REPO + shard
        n = struct.unpack("<Q", get(url, (0, 8)))[0]
        hdr = json.loads(get(url, (8, 8 + n)))
        base = 8 + n
        names = sorted(names, key=lambda k: hdr[k]["data_offsets"][0])
        sizes = [hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in names]
        print(f"{shard}: {len(names)} tensors, {sum(sizes) / 1e9:.2f} GB", flush=True)
        total += sum(sizes)

        def fetch(k):
            a, b = hdr[k]["data_offsets"]
            chunks = []
            step = 64 << 20
            for o in range(a, b, step):
                chunks.append(get(url, (base + o, base + min(b, o + step))))
            return b"".join(chunks)

        new_hdr, off = {}, 0
        for k, sz in zip(names, sizes):
            new_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + sz]}
            off += sz
        hj = json.dumps(new_hdr).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        tmp = dst + ".part"
        with open(tmp, "wb") as fo, cf.ThreadPoolExecutor(16) as ex:
            fo.write(struct.pack("<Q", len(hj)))
            fo.write(hj)
            for i in range(0, len(names), 64):
                for data in ex.map(fetch, names[i : i + 64]):
                    fo.write(data)
        os.rename(tmp, dst)
    print(f"done: {total / 1e9:.2f} GB fetched")


if __name__ == "__main__":
    main()
