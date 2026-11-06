"""Attention on Hopper: inputs and f64 reference for tang's derived flash
attention (`examples/hopper_attention.rs`), and the same tensors through
FlashAttention-3 and PyTorch SDPA.

    python scripts/hopper_attention.py gen DIR N D CAUSAL   # inputs + reference
    python scripts/hopper_attention.py bench DIR            # FA3 and SDPA

16k tokens per config (batch = 16384 / N) and 2048 hidden (heads = 2048 / D),
FlashAttention's benchmark convention. q, k, v, dO ~ N(0, 1) in bf16, layout
[batch * heads, N, D]. The reference is the f64 forward and backward of the
bf16 inputs for two heads (the first and the last), written as f32.

Timing: CUDA events, best of 5 samples of 20 launches (tang's harness does
the same). The forward does 4·N²·D FLOPs per head (half when causal), the
backward 2.5× that.
"""
import os
import sys

import torch
import torch.nn.functional as F

TOKENS, HIDDEN = 16384, 2048


def shape(n, d):
    return TOKENS // n, HIDDEN // d  # batch, heads


def gen(out, n, d, causal):
    os.makedirs(out, exist_ok=True)
    b, h = shape(n, d)
    bh = b * h
    g = torch.Generator(device="cuda").manual_seed(n * 1000 + d * 10 + causal)
    t = [torch.randn(bh, n, d, device="cuda", generator=g).bfloat16() for _ in range(4)]
    for name, x in zip(["q", "k", "v", "do"], t):
        x.contiguous().view(torch.int16).cpu().numpy().tofile(f"{out}/{name}.bin")
    heads = [0, bh - 1]
    q, k, v, do = (x[heads].double().requires_grad_(i < 3) for i, x in enumerate(t))
    s = q @ k.transpose(-1, -2) / d**0.5
    if causal:
        s = s.masked_fill(torch.ones(n, n, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    o = torch.softmax(s, -1) @ v
    dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
    for name, x in zip(["o", "dq", "dk", "dv"], [o, dq, dk, dv]):
        x.detach().float().cpu().numpy().tofile(f"{out}/{name}_ref.bin")
    with open(f"{out}/meta.txt", "w") as f:
        f.write(f"{bh} {n} {d} {int(causal)} {' '.join(map(str, heads))}\n")


def best_ms(fn, samples=5, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best


def rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


def load_fa3():
    """FlashAttention-3 (the Hopper build): the `flash_attn_interface`
    module from flash-attention's `hopper/`, else the prebuilt kernel from
    the Hugging Face kernels hub."""
    try:
        import flash_attn_interface as fa3

        return fa3.flash_attn_func, "flash_attn_interface"
    except ImportError:
        from kernels import get_kernel

        k = get_kernel("kernels-community/flash-attn3", version=1)
        return k.flash_attn_func, "kernels-community/flash-attn3"


def bench(out):
    bh, n, d, causal, *heads = map(int, open(f"{out}/meta.txt").read().split())
    causal = bool(causal)
    b, h = shape(n, d)

    def load(name):
        import numpy as np

        a = np.fromfile(f"{out}/{name}.bin", dtype=np.int16)
        return torch.from_numpy(a).cuda().view(torch.bfloat16).view(bh, n, d)

    q, k, v, do = (load(x) for x in ["q", "k", "v", "do"])
    ref = {}
    for name in ["o", "dq", "dk", "dv"]:
        import numpy as np

        ref[name] = torch.from_numpy(np.fromfile(f"{out}/{name}_ref.bin", dtype=np.float32)).cuda().view(len(heads), n, d)
    flop = 4.0 * bh * n * n * d * (0.5 if causal else 1.0)
    tag = f"bh={bh} n={n} d={d} causal={causal}"

    def run(name, fwd, to_bhnd):
        qq, kk, vv = (x.detach().clone().requires_grad_() for x in to_in(q, k, v))
        o = fwd(qq, kk, vv)
        oo = to_bhnd(o)
        e_o = rel(oo[heads], ref["o"])
        t_f = best_ms(lambda: fwd(qq, kk, vv))
        dd = to_out_like(do, o)
        grads = torch.autograd.grad(o, (qq, kk, vv), dd, retain_graph=True)
        errs = [rel(to_bhnd(g)[heads], ref[nm]) for g, nm in zip(grads, ["dq", "dk", "dv"])]
        t_b = best_ms(lambda: torch.autograd.grad(o, (qq, kk, vv), dd, retain_graph=True))
        print(f"{name} fwd {tag}: {t_f:.3f} ms {flop / t_f / 1e9:.1f} TFLOP/s rel_err {e_o:.2e}")
        print(
            f"{name} bwd {tag}: {t_b:.3f} ms {2.5 * flop / t_b / 1e9:.1f} TFLOP/s rel_err "
            + " ".join(f"{nm} {e:.2e}" for nm, e in zip(["dq", "dk", "dv"], errs))
        )

    # SDPA on [b, h, n, d]
    to_in = lambda *xs: [x.view(b, h, n, d) for x in xs]
    to_out_like = lambda x, o: x.view(b, h, n, d)
    run("sdpa", lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal), lambda o: o.reshape(bh, n, d))

    # FA3 on [b, n, h, d]
    fa3, src = load_fa3()
    print(f"# fa3 from {src}")
    to_in = lambda *xs: [x.view(b, h, n, d).transpose(1, 2).contiguous() for x in xs]
    to_out_like = lambda x, o: x.view(b, h, n, d).transpose(1, 2).contiguous()

    def fa3_fwd(q_, k_, v_):
        r = fa3(q_, k_, v_, causal=causal)
        return r[0] if isinstance(r, tuple) else r

    run("fa3", fa3_fwd, lambda o: o.transpose(1, 2).reshape(bh, n, d))
    if os.environ.get("ATT_FP8", "1") == "1":
        # FA3's FP8 (e4m3) forward, for reference: per-tensor scale 1
        try:
            q8, k8, v8 = (x.to(torch.float8_e4m3fn) for x in to_in(q, k, v))
            o8 = fa3_fwd(q8, k8, v8)
            e = rel(o8.float().transpose(1, 2).reshape(bh, n, d)[heads], ref["o"])
            t = best_ms(lambda: fa3_fwd(q8, k8, v8))
            print(f"fa3-fp8 fwd {tag}: {t:.3f} ms {flop / t / 1e9:.1f} TFLOP/s rel_err {e:.2e}")
        except Exception as ex:  # noqa: BLE001
            print(f"fa3-fp8 fwd {tag}: unavailable ({type(ex).__name__}: {ex})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "bench":
        bench(sys.argv[2])
    else:
        sys.exit(__doc__)
