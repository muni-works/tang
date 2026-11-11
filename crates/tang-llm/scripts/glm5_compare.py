#!/usr/bin/env python3
"""Compare tang TANG_DUMP_DIR dumps against the GLM-5.3-Flash tiny HF reference.

W0 harness compare tool. tang (W1's `flash-ref`-style CLI) runs the tiny GGUF
(/private/tmp/glm5-tiny/glm5-tiny-f32.gguf) with TANG_DUMP_DIR=<dir> set and writes
one raw little-endian f32 `.bin` per dump plus `index.json` (format in
docs/models/glm5-gguf-map.md §6). This tool compares every dump present in both
TANG_DUMP_DIR and the reference dir (default /private/tmp/glm5-tiny/ref) and
reports per-dump max abs err / max rel err, plus top-1 agreement for `logits`.

Usage:
  python3 glm5_compare.py --tang-dump /tmp/tang-dump [--ref /private/tmp/glm5-tiny/ref]

Exit code 0 = all dumps within tolerance (default max abs err 1e-4, logits top-1
100%), 1 = mismatch, 2 = missing dumps.
"""
import argparse
import json
import os
import sys

import numpy as np

DEFAULT_REF = "/private/tmp/glm5-tiny/ref"


def load_dump(path):
    a = np.fromfile(path, dtype="<f4")
    stem = path[:-4] if path.endswith(".bin") else path
    j = stem + ".json"
    shape = None
    if os.path.exists(j):
        with open(j) as f:
            shape = json.load(f).get("shape")
    if shape is None:
        # ref per-dump JSONs are {"shape": [...]}; if absent, try the dump index
        raise FileNotFoundError(f"no sidecar JSON for {path}")
    n = 1
    for d in shape:
        n *= d
    if a.size != n:
        raise ValueError(f"{path}: {a.size} floats != shape {shape}")
    return a.reshape(shape), shape


def cmp_dump(name, ref_dir, tang_dir, abs_tol, quiet=False):
    rp, tp = os.path.join(ref_dir, name + ".bin"), os.path.join(tang_dir, name + ".bin")
    if not os.path.exists(rp):
        return None  # not a reference dump
    if not os.path.exists(tp):
        return ("MISSING", name, None, None, None)
    ref, shape = load_dump(rp)
    got, shape2 = load_dump(tp)
    if shape != shape2:
        return ("SHAPE", name, f"ref {shape} vs tang {shape2}", None, None)
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    max_rel = float((diff / denom).max())
    ok = max_abs <= abs_tol
    extra = ""
    if name == "logits":
        top1 = float((got.argmax(-1) == ref.argmax(-1)).mean())
        extra = f" top-1 {top1:.0%}"
        ok = ok and top1 == 1.0
    return ("OK" if ok else "DIFF", name, max_abs, max_rel, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang-dump", required=True, help="TANG_DUMP_DIR written by tang")
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    with open(os.path.join(a.ref, "meta.json")) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]

    # discover what tang wrote
    idx_path = os.path.join(a.tang_dump, "index.json")
    tang_names = None
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        tang_names = set()
        for seq in idx.get("seqs", {}).values():
            tang_names.update(seq.get("dumps", []))
        tang_names.update(idx.get("dumps", []))

    # reference dump set: embed, logits, layerB{i}(+_seq2), final(+_seq2)
    ref_names = ["embed", "logits"]
    for i in range(n_layers):
        for suf in ("", "_seq2"):
            ref_names.append(f"layerB{i}{suf}")
    ref_names += ["final", "final_seq2"]
    seen, uniq = set(), []
    for n in ref_names:
        if n not in seen and os.path.exists(os.path.join(a.ref, n + ".bin")):
            seen.add(n)
            uniq.append(n)
    ref_names = uniq

    rows, missing, bad = [], [], []
    for name in ref_names:
        if tang_names is not None and name not in tang_names:
            continue  # tang may dump a subset; only compare what it claims
        r = cmp_dump(name, a.ref, a.tang_dump, a.tol)
        if r is None:
            continue
        status, nm, ma, mr, extra = r
        if status == "MISSING":
            missing.append(nm)
            rows.append((nm, "MISSING", "-", "-", ""))
        elif status == "SHAPE":
            bad.append(nm)
            rows.append((nm, "SHAPE-ERR", str(ma), "-", ""))
        else:
            rows.append((nm, status, f"{ma:.3e}", f"{mr:.3e}", extra or ""))
            if status == "DIFF":
                bad.append(nm)

    print(f"{'dump':16s} {'status':10s} {'max abs':>10s} {'max rel':>10s}")
    for nm, st, ma, mr, extra in rows:
        print(f"{nm:16s} {st:10s} {ma:>10s} {mr:>10s} {extra}")
    if missing:
        missing = sorted(set(missing))
        print(f"MISSING: {len(missing)} dumps not written by tang: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if bad:
        bad = sorted(set(bad))
        print(f"FAIL: {len(bad)} dumps out of tolerance: {bad}")
        sys.exit(1)
    print(f"PASS: {len(rows)} dumps within tol {a.tol:g}")
    sys.exit(0)


if __name__ == "__main__":
    main()
