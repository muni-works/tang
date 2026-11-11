#!/usr/bin/env python3
"""Requantize GLM-5.3-Flash's KDA projection tensors, leave every other byte as it is.

    glm5-kda-requant.py SRC_DIR DST_DIR {q4_k|q5_k} --ggml /path/libggml-base.so

For each `*.gguf` shard in SRC_DIR: the KDA layers' (those with `blk.N.attn_q.weight`)
attn_q / attn_k / attn_v / attn_output / ssm_f_a / ssm_g_a, stored Q8_0, are dequantized
(gguf-py) and requantized with ggml's own `ggml_quantize_chunk` (ctypes, no imatrix). All
metadata and all other tensors are copied verbatim. A shard with no target tensor is
symlinked, not copied. The MLA layers' attn_output (16384 wide) is not touched.

Needs gguf-py on PYTHONPATH (llama.cpp/gguf-py) and a built libggml-base.so.
"""
import argparse
import ctypes
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import gguf

TARGETS = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
           "ssm_f_a.weight", "ssm_g_a.weight")
TYPES = {"q4_k": gguf.GGMLQuantizationType.Q4_K, "q5_k": gguf.GGMLQuantizationType.Q5_K}


def kda_layers(paths):
    out = set()
    for p in paths:
        for t in gguf.GGUFReader(p).tensors:
            m = re.match(r"blk\.(\d+)\.attn_q\.weight$", t.name)
            if m:
                out.add(int(m.group(1)))
    return out


def is_target(name, layers):
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    return bool(m) and int(m.group(1)) in layers and m.group(2) in TARGETS


def requant(lib, qtype, t):
    assert t.tensor_type == gguf.GGMLQuantizationType.Q8_0, (t.name, t.tensor_type)
    w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
    w = np.ascontiguousarray(w.reshape(-1, w.shape[-1]))
    rows, cols = w.shape
    bs, ts = gguf.GGML_QUANT_SIZES[qtype]
    assert cols % bs == 0, (t.name, cols)
    rb = cols // bs * ts
    dst = np.empty((rows, rb), dtype=np.uint8)
    chunk = max(1, rows // 32)

    def run(r0):
        n = min(chunk, rows - r0)
        got = lib.ggml_quantize_chunk(int(qtype), w.ctypes.data_as(ctypes.c_void_p),
                                      dst.ctypes.data_as(ctypes.c_void_p), r0 * cols, n, cols, None)
        assert got == n * rb, (t.name, got, n * rb)

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(run, range(0, rows, chunk)))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("type", choices=TYPES)
    ap.add_argument("--ggml", required=True)
    a = ap.parse_args()
    lib = ctypes.CDLL(a.ggml)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    qtype = TYPES[a.type]
    srcs = sorted(Path(a.src).glob("*.gguf"))
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    layers = kda_layers(srcs)
    print(f"{len(layers)} KDA layers: {sorted(layers)}", flush=True)
    tot_old = tot_new = 0
    for p in srcs:
        r = gguf.GGUFReader(p)
        out = dst / p.name
        if not any(is_target(t.name, layers) for t in r.tensors):
            if out.exists() or out.is_symlink():
                out.unlink()
            os.symlink(p.resolve(), out)
            print(f"{p.name}: no targets, symlinked", flush=True)
            continue
        w = gguf.GGUFWriter(out, arch="", endianess=r.endianess)
        # the writer adds general.architecture itself; shards 2.. don't carry it: drop it and
        # copy exactly the source's KV set
        for kv in w.kv_data:
            kv.pop(gguf.Keys.General.ARCHITECTURE, None)
        for f in r.fields.values():
            if f.name.startswith("GGUF."):
                continue
            vt = f.types[0]
            st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, f.contents(), vt, sub_type=st)
        new = {}
        for t in r.tensors:
            if is_target(t.name, layers):
                d = requant(lib, qtype, t)
                new[t.name] = d
                tot_old += t.n_bytes
                tot_new += d.nbytes
                w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, qtype)
            else:
                w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        print(f"{p.name}: {len(new)} tensors requantized", flush=True)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in r.tensors:
            w.write_tensor_data(new.pop(t.name, t.data), tensor_endianess=r.endianess)
        w.close()
    print(f"targets: {tot_old / 1e9:.3f} GB Q8_0 -> {tot_new / 1e9:.3f} GB {a.type}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
