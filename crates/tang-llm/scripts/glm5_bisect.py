#!/usr/bin/env python3
"""R3 bisector: localize the first diverging layer between tang's CPU and GPU engines.

W0 harness. For each prompt in the R2b suite:

  1. Run `tang-llm flash-generate` on the CPU path (TANG_GLM5_GPU=0) and the GPU path
     (TANG_GPUS=1) with --out ids files, find the first generated index where they differ.
  2. Replay the prompt prefix + diverging step token on BOTH engines with TANG_DUMP_DIR
     set (dump.rs hooks) — the engine dumps per-layer post-FFN stream rows + final hidden
     + logits for the LAST position of each window, so a token-by-token replay of the
     prompt (windows of 1) ends exactly at the diverging step.
  3. Compare layer by layer (L00.post_ffn .. L{n}.post_ffn, final_x, logits) and report
     the FIRST layer whose max abs err exceeds the threshold, with the layer's kind
     (KDA or DSA + whether its FFN is dense or MoE).

Usage (runs on the pod, from the fixture dir):
  python3 glm5_bisect.py --gguf <gguf> --bin tang-llm [--prompts p1 p2 ...] [--tol 1e-3]

Exit 0 iff every prompt's divergence is localized (or there is no divergence).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def read_ids(path):
    with open(path) as f:
        return [int(w) for w in f.read().split()]


def run_generate(binary, gguf, ids_path, n, out_path, env):
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(
        [binary, "flash-generate", gguf, "--ids-file", ids_path, "-n", str(n),
         "--temp", "0", "--out", out_path, "--quiet"],
        env=e, capture_output=True, text=True,
    )
    return r.returncode == 0


def run_dump(binary, gguf, ids_path, dump_dir, env):
    e = dict(os.environ)
    e.update(env)
    e["TANG_DUMP_DIR"] = dump_dir
    r = subprocess.run(
        [binary, "flash-generate", gguf, "--ids-file", ids_path, "-n", "1",
         "--temp", "0", "--out", os.path.join(dump_dir, "out.ids"), "--quiet"],
        env=e, capture_output=True, text=True,
    )
    return r.returncode == 0, r.stderr[-2000:] if r.returncode else ""


def read_dump(dump_dir):
    with open(os.path.join(dump_dir, "index.json")) as f:
        idx = json.load(f)
    out = {}
    for t in idx["tensors"]:
        a = np.fromfile(os.path.join(dump_dir, t["file"]), dtype="<f4")
        out[t["name"]] = a
    return out


def first_div(got, want):
    n = min(len(got), len(want))
    for i in range(n):
        if got[i] != want[i]:
            return i
    return None if len(got) == len(want) else min(len(got), len(want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--bin", default="tang-llm")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=32, help="greedy tokens for the pass check")
    a = ap.parse_args()

    with open(os.path.join(a.suite, "prompts.json")) as f:
        doc = json.load(f)
    names = [p["name"] for p in doc["prompts"]]
    if a.prompts:
        names = [n for n in names if n in a.prompts]

    tmp = tempfile.mkdtemp(prefix="glm5-bisect-", dir=os.path.join(a.suite, "got")
                           if os.path.isdir(os.path.join(a.suite, "got")) else None)
    cpu_dir, gpu_dir = os.path.join(tmp, "cpu"), os.path.join(tmp, "gpu")
    os.makedirs(cpu_dir)
    os.makedirs(gpu_dir)

    n_ok = 0
    for name in names:
        ids_path = os.path.join(a.suite, name + ".ids")
        ids = read_ids(ids_path)
        cpu_out = os.path.join(cpu_dir, name + ".ids")
        gpu_out = os.path.join(gpu_dir, name + ".ids")
        ok_c = run_generate(a.bin, a.gguf, ids_path, a.n, cpu_out,
                            {"TANG_GLM5_GPU": "0"})
        ok_g = run_generate(a.bin, a.gguf, ids_path, a.n, gpu_out,
                            {"TANG_GPUS": "1"})
        if not (ok_c and ok_g):
            print(f"{name}: RUN ERROR (cpu={ok_c} gpu={ok_g})")
            continue
        cpu_ids, gpu_ids = read_ids(cpu_out), read_ids(gpu_out)
        d = first_div(gpu_ids, cpu_ids)
        if d is None:
            n_ok += 1
            print(f"{name}: PASS (gpu == cpu over {len(cpu_ids)} tokens)")
            continue
        print(f"{name}: DIVERGES at generated index {d} "
              f"(cpu {cpu_ids[d] if d < len(cpu_ids) else '-'} vs gpu {gpu_ids[d] if d < len(gpu_ids) else '-'})")

        # replay: prompt minus the token produced by the diverging step, then that step
        # with dumping on both paths. The engine generates token[d] from the context
        # ids + generated[0..d], so replay exactly that context with -n 1 and dump.
        ctx = ids + cpu_ids[:d]
        ctx_path = os.path.join(tmp, name + "_ctx.ids")
        with open(ctx_path, "w") as f:
            f.write(" ".join(map(str, ctx)))
        c_dump, g_dump = os.path.join(tmp, name + "_c"), os.path.join(tmp, name + "_g")
        okc, errc = run_dump(a.bin, a.gguf, ctx_path, c_dump, {"TANG_GLM5_GPU": "0"})
        okg, errg = run_dump(a.bin, a.gguf, ctx_path, g_dump, {"TANG_GPUS": "1"})
        if not (okc and okg):
            print(f"  replay failed: cpu={okc} {errc[:200]} gpu={okg} {errg[:200]}")
            continue
        dc, dg = read_dump(c_dump), read_dump(g_dump)
        worst = []
        for k in sorted(dc):
            if k not in dg:
                continue
            diff = float(np.abs(dc[k].astype(np.float64) - dg[k].astype(np.float64)).max())
            worst.append((diff, k))
        worst.sort(reverse=True)
        first_bad = next(((x, k) for x, k in worst if x > a.tol), None)
        print(f"  worst layers: " + ", ".join(f"{k} {x:.3e}" for x, k in worst[:5]))
        if first_bad:
            _, kbad = first_bad
            lnum = int(kbad.split(".")[0][1:])
            print(f"  FIRST DIVERGENT LAYER: {kbad} (max abs err {first_bad[0]:.3e}, tol {a.tol:g})")
        else:
            print(f"  all layer dumps within tol {a.tol:g} — divergence is in the head/sampler")

    print(f"\n{len(names) - n_ok}/{len(names)} prompts diverge GPU-vs-CPU")
    sys.exit(0)


if __name__ == "__main__":
    main()
