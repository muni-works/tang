"""Baselines + correctness for tang's derived attention variants.

Reads what `cargo run --example attn_variants -- OUT` wrote (inputs and tang's
outputs per variant/d/n/pass), checks tang against an f64 torch reference, and
runs the baselines on the same inputs:

- FlexAttention (torch.compile'd flex_attention, score_mod + block_mask), fp32
  with TF32 matmuls allowed (tang's tensor-core tiles are TF32 too), and bf16;
- FlashAttention-2 (flash_attn package) in bf16 where the variant is supported
  (causal, sliding window, ALiBi, softcap, bidirectional; prefix LM is not).
- PyTorch SDPA with an explicit additive mask (math path), fp32, for context.

Shape: B = 1, H = 1, [n, d] -- the shape tang's naive_graph builds (it has no
head dim), so every system sees exactly the same problem.

usage: python scripts/attn_variants_bench.py OUT tang.csv > results.csv
"""
import argparse, csv, math, os, sys
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
flex = torch.compile(flex_attention, dynamic=False)

SLOPE = 0.0625
# (score_mod, mask_mod, dense keep(i,j), dense score(s,i,j), fa2 kwargs or None)
VARIANTS = {
    "causal": (None, lambda b, h, i, j: j <= i, lambda i, j: j <= i, lambda s, i, j: s, dict(causal=True)),
    "sliding128": (None, lambda b, h, i, j: (j <= i) & (i - j < 128), lambda i, j: (j <= i) & (i - j < 128),
                   lambda s, i, j: s, dict(causal=True, window_size=(127, 0))),
    "alibi": (lambda s, b, h, i, j: s - SLOPE * (i - j), lambda b, h, i, j: j <= i, lambda i, j: j <= i,
              lambda s, i, j: s - SLOPE * (i - j), "alibi"),
    "softcap": (lambda s, b, h, i, j: 30 * torch.tanh(s / 30), lambda b, h, i, j: j <= i, lambda i, j: j <= i,
                lambda s, i, j: 30 * torch.tanh(s / 30), dict(causal=True, softcap=30.0)),
    "prefix64": (None, lambda b, h, i, j: (j <= i) | (j < 64), lambda i, j: (j <= i) | (j < 64),
                 lambda s, i, j: s, None),
    "bidirectional": (None, None, None, lambda s, i, j: s, dict(causal=False)),
}


def density(name, n):
    i = torch.arange(n, device=dev)[:, None]
    j = torch.arange(n, device=dev)[None, :]
    keep = VARIANTS[name][2]
    return 1.0 if keep is None else keep(i, j).float().mean().item()


def reference(name, q, k, v, do):
    """f64 dense reference; returns O or (dq, dk, dv)."""
    n, d = q.shape
    q, k, v = (x.double().requires_grad_(do is not None) for x in (q, k, v))
    i = torch.arange(n, device=dev)[:, None].double()
    j = torch.arange(n, device=dev)[None, :].double()
    s = VARIANTS[name][3](q @ k.T / math.sqrt(d), i, j)
    keep = VARIANTS[name][2]
    if keep is not None:
        s = s.masked_fill(~keep(i, j), float("-inf"))
    o = torch.softmax(s, -1) @ v
    if do is None:
        return [o.detach()]
    return list(torch.autograd.grad(o, (q, k, v), do.double()))


def time_ms(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):  # min over 5 batches, as tang's Program::time
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


def rel_err(got, ref):
    got = torch.cat([g.reshape(-1).double() for g in got])
    ref = torch.cat([r.reshape(-1) for r in ref])
    return ((got - ref).abs().max() / ref.abs().max()).item()


def run_flex(name, q, k, v, do, dtype, bm_cache):
    n, d = q.shape
    smod, mmod = VARIANTS[name][0], VARIANTS[name][1]
    key = (name, n)
    if mmod is not None and key not in bm_cache:
        bm_cache[key] = create_block_mask(mmod, 1, 1, n, n, device=dev)
    bm = bm_cache.get(key) if mmod is not None else None
    H = q.shape[1] if q.dim() == 4 else 1
    if q.dim() == 2:
        q, k, v = (x.to(dtype)[None, None] for x in (q, k, v))
    bwd = do is not None
    q, k, v = (x.detach().requires_grad_(bwd) for x in (q, k, v))
    f = lambda: flex(q, k, v, score_mod=smod, block_mask=bm)
    if not bwd:
        return f, lambda: [f()[0, 0]]
    dov = do.to(dtype).expand_as(q) if do.dim() == 4 else do.to(dtype)[None, None]

    def fb():
        o = f()
        return torch.autograd.grad(o, (q, k, v), dov)
    return fb, lambda: [g[0, 0] for g in fb()]


def run_fa2(name, q, k, v, do):
    kw = VARIANTS[name][4]
    if flash_attn_func is None or kw is None:
        return None
    n, d = q.shape[-2:]
    if q.dim() == 2:
        q, k, v = (x.bfloat16()[None, :, None, :] for x in (q, k, v))  # [B, n, H, d]
    else:
        q, k, v = (x.bfloat16().transpose(1, 2).contiguous() for x in (q, k, v))
    if kw == "alibi":
        # flash-attn's ALiBi adds -slope*|i-j|; with causal that's -slope*(i-j)
        kw = dict(causal=True, alibi_slopes=torch.full((q.shape[2],), SLOPE, device=dev, dtype=torch.float32))
    bwd = do is not None
    q, k, v = (x.detach().requires_grad_(bwd) for x in (q, k, v))
    f = lambda: flash_attn_func(q, k, v, **kw)
    if not bwd:
        return f, lambda: [f()[0, :, 0]]
    dov = do.bfloat16()[None, :, None, :].expand_as(q).contiguous()

    def fb():
        return torch.autograd.grad(f(), (q, k, v), dov)
    return fb, lambda: [g[0, :, 0] for g in fb()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("tang_csv")
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name()
    w = csv.writer(sys.stdout)
    w.writerow(["variant", "d", "n", "pass", "system", "heads", "ms", "tflops", "rel_err", "note"])
    bm_cache = {}
    for row in csv.DictReader(open(a.tang_csv)):
        name, d, n, p = row["variant"], int(row["d"]), int(row["n"]), row["pass"]
        bwd = p == "fwdbwd"
        tag = f"{name}_d{d}_n{n}_{p}"
        flops = 4 * n * n * d * density(name, n) * (3.5 if bwd else 1.0)
        tf = lambda ms, h=1: flops * h / (ms * 1e-3) / 1e12
        inb = os.path.join(a.out, tag + ".in.bin")
        if row["ms"] == "FAIL" or not os.path.exists(inb):
            w.writerow([name, d, n, p, "tang", 1, "", "", "", "compile/run failed"])
            continue
        ins = torch.from_file(inb, size=(4 if bwd else 3) * n * d, dtype=torch.float32).to(dev).view(-1, n, d)
        q, k, v = ins[0], ins[1], ins[2]
        do = ins[3] if bwd else None
        ref = reference(name, q, k, v, do)
        tout = torch.from_file(os.path.join(a.out, tag + ".out.bin"), size=(3 if bwd else 1) * n * d,
                               dtype=torch.float32).to(dev).view(-1, n, d)
        ms = float(row["ms"])
        w.writerow([name, d, n, p, "tang", 1, f"{ms:.4f}", f"{tf(ms):.2f}", f"{rel_err(list(tout), ref):.2e}",
                    f"kernels={row['kernels']} nxn={row['nxn_buffers']} mma={row['native_kept']}/{row['native_total']}"])
        iters = 10 if n >= 4096 else 30
        systems = [("flex_fp32", lambda: run_flex(name, q, k, v, do, torch.float32, bm_cache)),
                   ("flex_bf16", lambda: run_flex(name, q, k, v, do, torch.bfloat16, bm_cache)),
                   ("fa2_bf16", lambda: run_fa2(name, q, k, v, do))]
        for sysname, mk in systems:
            try:
                r = mk()
                if r is None:
                    w.writerow([name, d, n, p, sysname, 1, "", "", "", "unsupported"])
                    continue
                f, outs = r
                err = rel_err(outs(), ref)
                ms = time_ms(f, iters)
                w.writerow([name, d, n, p, sysname, 1, f"{ms:.4f}", f"{tf(ms):.2f}", f"{err:.2e}", ""])
            except Exception as e:
                w.writerow([name, d, n, p, sysname, 1, "", "", "", f"error: {type(e).__name__}: {str(e)[:80]}"])
            sys.stdout.flush()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
