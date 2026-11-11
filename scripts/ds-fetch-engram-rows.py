#!/usr/bin/env python3
"""Fetch only the Engram table rows a prompt hits, by HTTP range requests.

The Engram tables (layers.{1,14}.engram.embed.weight, ~100 GB each, fp8 rows of
engram_head_dim with one e8m0 scale per 32) stay on the host in a real run; a
CPU check needs only the rows its tokens' n-gram hashes name. The hashes are
computed here by DeepSeek-V4.1's reference (`deepseek_ref.EngramHash`, torch),
independently of tang's Rust hashing, which must then find every row it wants.

    ds-fetch-engram-rows.py OUT_DIR REF_DIR IDS LAYER

OUT_DIR holds inference_config.json and engram.json (from
deepseek_engram_meta.py); REF_DIR holds deepseek_ref.py; IDS is a comma list of
checkpoint token ids (the whole sequence); LAYER a checkpoint Engram layer.
Writes OUT_DIR/engram-rows-l{LAYER}.safetensors: the rows (sorted unique) as
`layers.{L}.engram.embed.weight` [n, ehd] F8_E4M3 and `.scale` [n, ehd/32]
F8_E8M0, plus `layers.{L}.engram.row_ids` [n] I64.
"""
import json
import os
import struct
import sys
import urllib.request

import numpy as np
import torch

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"


def get(url, rng):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != rng[1] - rng[0]:
                raise IOError(f"short read {len(data)}")
            return data
        except Exception as e:
            if attempt == 5:
                raise
            print(f"retry {rng}: {e}", file=sys.stderr)


def main():
    out, ref_dir, ids, layer = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, ref_dir)
    import deepseek_ref as R

    c = json.load(open(os.path.join(out, "inference_config.json")))
    meta = json.load(open(os.path.join(out, "engram.json")))
    c["max_seq_len"] = 1 << 16
    ids = torch.tensor([int(x) for x in ids.split(",")])
    h = R.EngramHash(c, meta["token_map"], c["engram_compressed_vocab_size"])
    j = c["engram_layer_ids"].index(layer)
    rows = sorted(set(h(ids, 0)[:, j].flatten().tolist()))
    print(f"layer {layer}: {len(ids)} tokens hit {len(rows)} rows")
    wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    name = f"layers.{layer}.engram.embed.weight"
    sname = f"layers.{layer}.engram.embed.scale"
    url = REPO + wm[name]
    n = struct.unpack("<Q", get(url, (0, 8)))[0]
    hdr = json.loads(get(url, (8, 8 + n)))
    base = 8 + n
    ehd = hdr[name]["shape"][1]
    sc = hdr[sname]["shape"][1]
    assert hdr[name]["dtype"] == "F8_E4M3" and hdr[sname]["dtype"] == "F8_E8M0"
    w, s = b"", b""
    for r in rows:
        a = base + hdr[name]["data_offsets"][0] + r * ehd
        w += get(url, (a, a + ehd))
        a = base + hdr[sname]["data_offsets"][0] + r * sc
        s += get(url, (a, a + sc))
    rid = np.array(rows, dtype=np.int64).tobytes()
    k = len(rows)
    new = {
        name: {"dtype": "F8_E4M3", "shape": [k, ehd], "data_offsets": [0, len(w)]},
        sname: {"dtype": "F8_E8M0", "shape": [k, sc], "data_offsets": [len(w), len(w) + len(s)]},
        f"layers.{layer}.engram.row_ids": {
            "dtype": "I64",
            "shape": [k],
            "data_offsets": [len(w) + len(s), len(w) + len(s) + len(rid)],
        },
    }
    hj = json.dumps(new).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    dst = os.path.join(out, f"engram-rows-l{layer}.safetensors")
    with open(dst + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + w + s + rid)
    os.rename(dst + ".part", dst)
    print(f"-> {dst} ({len(w) + len(s)} bytes of table)")


if __name__ == "__main__":
    main()
