#!/usr/bin/env python3
"""Fetch a few layers of GLM-5.3-Flash (unsloth/GLM-5.3-Flash-GGUF, UD-Q4_K_XL)
by HTTP range requests into one small GGUF.

    glm-fetch-layers.py OUT.gguf LAYERS        # e.g. 0,3
    glm-fetch-layers.py --plan [LAYERS]        # layer kinds, sizes; fetch nothing
    GLM_SRC=DIR glm-fetch-layers.py ...         # cut from a local copy instead

The output keeps the first shard's metadata verbatim (so `Hparams::from_gguf`
reads the real model's hparams) and holds `token_embd`, `output_norm`,
`output` and the wanted trunk layers, renamed in order: checkpoint layer
LAYERS[i] becomes `blk.{i}`. Tensor bytes are copied untouched (no
requantization). The Rust side remaps the per-layer hparams to match.
"""
import concurrent.futures as cf
import json
import os
import struct
import sys
import urllib.request

# GLM_SRC=DIR reads a local copy of the shards instead (same layout)
REPO = os.environ.get("GLM_SRC", "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-Q4_K_XL")
REPO = REPO.rstrip("/") + "/"
SHARDS = [f"GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf" for i in range(1, 7)]


def get(url, rng=None, short_ok=False):
    if not url.startswith("http"):
        with open(url, "rb") as f:
            if not rng:
                return f.read()
            f.seek(rng[0])
            return f.read(rng[1] - rng[0])
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
            if rng and len(data) != rng[1] - rng[0] and not short_ok:
                raise IOError(f"short read {len(data)}")
            return data
        except Exception as e:
            if attempt == 5:
                raise
            print(f"retry {url} {rng}: {e}", file=sys.stderr)


def size_of(url):
    if not url.startswith("http"):
        return os.path.getsize(url)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class Rd:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0

    def need(self, n):
        while self.pos + n > len(self.buf):
            a = len(self.buf)
            b = get(self.url, (a, a + (4 << 20)), short_ok=True)
            if not b:
                raise EOFError(self.url)
            self.buf += b

    def take(self, n):
        self.need(n)
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64()).decode()


SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def value(r, t):
    if t == 8:
        return r.s()
    if t == 9:
        et, n = r.u32(), r.u64()
        return [value(r, et) for _ in range(n)]
    return struct.unpack("<" + FMT[t], r.take(SCALAR[t]))[0]


def header(url):
    r = Rd(url)
    assert r.take(4) == b"GGUF"
    ver, nt, nkv = r.u32(), r.u64(), r.u64()
    meta, kvs = {}, []
    for _ in range(nkv):
        a = r.pos
        k = r.s()
        t = r.u32()
        et = None
        if t == 9:
            r.need(4)
            et = struct.unpack("<I", r.buf[r.pos : r.pos + 4])[0]
        meta[k] = value(r, t)
        kvs.append((k, t, et, r.buf[a : r.pos]))
    kv_bytes = kvs
    tensors = []
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty, off = r.u32(), r.u64()
        tensors.append((name, dims, ty, off))
    align = meta.get("general.alignment", 32)
    data = (r.pos + align - 1) // align * align
    return ver, nkv, kv_bytes, meta, tensors, data, align


def main():
    args = sys.argv[1:]
    plan = args and args[0] == "--plan"
    if plan:
        args = args[1:]
    out = None if plan else args.pop(0)
    layers = [int(x) for x in (args[0] if args else "0").split(",")]
    hs = []
    with cf.ThreadPoolExecutor(6) as ex:
        hs = list(ex.map(lambda s: (s, header(REPO + s), size_of(REPO + s)), SHARDS))
    meta = hs[0][1][3]
    arch = meta["general.architecture"]
    kvh = meta[f"{arch}.attention.head_count_kv"]
    nl = meta[f"{arch}.block_count"]
    lead = meta.get(f"{arch}.leading_dense_block_count", 0)
    if plan:
        for k, v in meta.items():
            if k.startswith(arch) or k.startswith("split"):
                print(k, v if not isinstance(v, list) or len(v) < 64 else f"[{len(v)} items]")
    # every tensor: (shard, name, dims, type, abs start, size)
    allt = []
    for shard, (ver, nkv, kvb, m, ts, data, align), fsize in hs:
        order = sorted(ts, key=lambda t: t[3])
        for i, (name, dims, ty, off) in enumerate(order):
            end = order[i + 1][3] if i + 1 < len(order) else fsize - data
            allt.append((shard, name, dims, ty, data + off, end - off))
    per_layer = {}
    for t in allt:
        if t[1].startswith("blk."):
            per_layer.setdefault(int(t[1].split(".")[1]), []).append(t)
    if plan:
        for l in range(nl):
            ts = per_layer.get(l, [])
            kind = "kda" if isinstance(kvh, list) and kvh[l] == 0 else "dsa"
            ffn = "dense" if l < lead else "moe"
            idx = any(".indexer." in t[1] for t in ts)
            print(f"layer {l:2} {kind} {ffn}{' indexer' if idx else ''}: {sum(t[5] for t in ts) / 1e9:.2f} GB, types {sorted(set(t[3] for t in ts))}")
    head = [t for t in allt if t[1] in ("token_embd.weight", "output_norm.weight", "output.weight")]
    want = head + [t for l in layers for t in per_layer[l]]
    total = sum(t[5] for t in want)
    print(f"layers {layers} + embed/head: {len(want)} tensors, {total / 1e9:.2f} GB")
    if plan:
        return
    rename = {l: i for i, l in enumerate(layers)}

    def newname(n):
        if not n.startswith("blk."):
            return n
        p = n.split(".")
        p[1] = str(rename[int(p[1])])
        return ".".join(p)

    ver, nkv, kvs, m, ts, data, align = hs[0][1]
    # a self-consistent len(layers)-layer model: per-layer arrays cut to the
    # chosen layers, no NextN, no split keys
    def enc(t, v):
        if t == 8:
            b = v.encode()
            return struct.pack("<Q", len(b)) + b
        return struct.pack("<" + FMT[t], v)

    def kv(k, t, et, v):
        kb = k.encode()
        out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t)
        if t == 9:
            return out + struct.pack("<IQ", et, len(v)) + b"".join(enc(et, x) for x in v)
        return out + enc(t, v)

    n_lead = sum(1 for l in layers if l < lead)
    assert layers[:n_lead] == sorted(l for l in layers if l < lead), "dense layers first"
    per_layer = {f"{arch}.attention.head_count_kv", f"{arch}.swiglu_clamp_exp",
                 f"{arch}.swiglu_clamp_shexp", f"{arch}.attention.indexer.types"}
    kvb, nkv = b"", 0
    for k, t, et, raw in kvs:
        if k.startswith("split."):
            continue
        if k in per_layer and t == 9:
            raw = kv(k, t, et, [m[k][l] for l in layers])
        elif k == f"{arch}.block_count":
            raw = kv(k, t, et, len(layers))
        elif k == f"{arch}.nextn_predict_layers":
            raw = kv(k, t, et, 0)
        elif k == f"{arch}.leading_dense_block_count":
            raw = kv(k, t, et, n_lead)
        kvb += raw
        nkv += 1
    infos, off = b"", 0
    placed = []
    for t in want:
        nm = newname(t[1]).encode()
        infos += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(t[2]))
        infos += b"".join(struct.pack("<Q", d) for d in t[2])
        infos += struct.pack("<IQ", t[3], off)
        placed.append((t, off))
        off += (t[5] + align - 1) // align * align
    hdr = b"GGUF" + struct.pack("<IQQ", ver, len(want), nkv) + kvb + infos
    hdr += b"\0" * ((align - len(hdr) % align) % align)

    def fetch(t):
        shard, _, _, _, a, sz = t
        step = 64 << 20
        return b"".join(get(REPO + shard, (o, min(a + sz, o + step))) for o in range(a, a + sz, step))

    tmp = out + ".part"
    with open(tmp, "wb") as fo, cf.ThreadPoolExecutor(16) as ex:
        fo.write(hdr)
        base = len(hdr)
        for (t, o), b in zip(placed, ex.map(lambda p: fetch(p[0]), placed)):
            fo.seek(base + o)
            fo.write(b)
            print(f"  {t[1]} -> {newname(t[1])} {len(b) / 1e6:.1f} MB", flush=True)
        fo.truncate(base + off)
    os.rename(tmp, out)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
