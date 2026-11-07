"""Attention on Hopper: inputs and f64 reference for tang's derived flash
attention (`examples/hopper_attention.rs`), and the same tensors through
FlashAttention-3 and PyTorch SDPA.

    python scripts/hopper_attention.py gen DIR N D CAUSAL   # inputs + reference
    python scripts/hopper_attention.py bench DIR            # FA3 and SDPA
    python scripts/hopper_attention.py matmul [MxKxN,...]   # cuBLAS bf16
    python scripts/hopper_attention.py fp8mm [MxKxN,...]    # cuBLASLt FP8
    python scripts/hopper_attention.py fp8attn [N,...] [CAUSAL]  # FA3 FP8 / bf16, SDPA

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


def matmul(cases):
    """cuBLAS (torch bf16 matmul, fp32 accumulation) on the shapes of
    `examples/hopper_matmul.rs`, both B layouts."""
    for m, k, n in cases:
        for weight in (False, True):
            a = torch.randn(m, k, device="cuda").bfloat16()
            b = torch.randn(n, k, device="cuda").bfloat16() if weight else torch.randn(k, n, device="cuda").bfloat16()
            f = (lambda: a @ b.t()) if weight else (lambda: a @ b)
            t = best_ms(f)
            lay = "[n,k]" if weight else "[k,n]"
            print(f"cublas matmul {m}x{k}x{n} b={lay}: {t:.4f} ms {2 * m * k * n / t / 1e9:.1f} TFLOP/s")


def fp8mm(cases):
    """cuBLASLt FP8 (`torch._scaled_mm`, e4m3 × e4m3, fp32 accumulation,
    bf16 out) on the shapes of `examples/hopper_fp8.rs`: A [m, k] and B
    [n, k], per-tensor and per-row/column ("rowwise") scales. cuBLASLt on
    sm_90 has no 1×128 block scaling; tang's tiles do (DeepSeek-style)."""
    for m, k, n in cases:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(n, k, device="cuda")
        fa = a.to(torch.float8_e4m3fn)
        fb = b.to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")
        f = lambda: torch._scaled_mm(fa, fb.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        t = best_ms(f)
        line = f"cublaslt fp8 {m}x{k}x{n}: tensor {t:.4f} ms {2 * m * k * n / t / 1e9:.1f} TFLOP/s"
        try:
            sa = torch.ones(m, 1, device="cuda")
            sb = torch.ones(1, n, device="cuda")
            g = lambda: torch._scaled_mm(fa, fb.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            t2 = best_ms(g)
            line += f"  rowwise {t2:.4f} ms {2 * m * k * n / t2 / 1e9:.1f} TFLOP/s"
        except Exception as e:  # noqa: BLE001
            line += f"  rowwise n/a ({type(e).__name__})"
        print(line)


def fp8attn(ns, causal):
    """FlashAttention-3 FP8 forward (e4m3 Q, K, V with per-(batch, head)
    descales, bf16 out) and FA3 / SDPA bf16 on the shapes of
    `examples/hopper_fp8_attn.rs`: 16k tokens, 16 heads of 128."""
    try:
        fa3, src = load_fa3()
        print(f"# FA3 from {src}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# FA3 unavailable ({type(e).__name__}: {e})", flush=True)
        fa3 = None
    for n in ns:
        b, h, d = 16384 // n, 16, 128
        q, k, v = (torch.randn(b, n, h, d, device="cuda") for _ in range(3))
        flop = 4 * b * h * n * n * d * (0.5 if causal else 1.0)
        line = f"n={n} d={d} bh={b * h} causal={causal}:"
        qb, kb, vb = (x.bfloat16() for x in (q, k, v))
        t = best_ms(lambda: F.scaled_dot_product_attention(qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2), is_causal=causal))
        line += f"  sdpa bf16 {t:.3f} ms {flop / t / 1e9:.1f} TFLOP/s"
        if fa3 is not None:
            t = best_ms(lambda: fa3(qb, kb, vb, causal=causal))
            line += f"  fa3 bf16 {t:.3f} ms {flop / t / 1e9:.1f} TFLOP/s"
            amax = [x.abs().amax(dim=(1, 3)) for x in (q, k, v)]  # [b, h]
            sc = [(a / 448.0).float() for a in amax]
            q8, k8, v8 = (
                (x / s[:, None, :, None]).to(torch.float8_e4m3fn) for x, s in zip((q, k, v), sc)
            )
            f = lambda: fa3(q8, k8, v8, causal=causal, q_descale=sc[0], k_descale=sc[1], v_descale=sc[2])
            try:
                t = best_ms(f)
                o = f()
                o = o[0] if isinstance(o, tuple) else o
                # error vs f64 on the first and last head of batch 0
                hs = [0, h - 1]
                qd, kd, vd = (x[0, :, hs].double().transpose(0, 1) for x in (q, k, v))
                ref = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal)
                got = o[0, :, hs].double().transpose(0, 1)
                err = ((got - ref).abs().amax() / ref.abs().amax()).item()
                mean = ((got - ref).abs().mean() / ref.abs().amax()).item()
                line += f"  fa3 fp8 {t:.3f} ms {flop / t / 1e9:.1f} TFLOP/s (max_rel {err:.2e} mean_rel {mean:.2e})"
            except Exception as e:  # noqa: BLE001
                line += f"  fa3 fp8 n/a ({type(e).__name__}: {e})"
        print(line, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "bench":
        bench(sys.argv[2])
    elif cmd == "matmul":
        spec = sys.argv[2] if len(sys.argv) > 2 else "1024x1024x1024,2048x2048x2048,4096x4096x4096,8192x8192x8192,16384x2048x2048,16384x2048x8192,16384x8192x2048"
        matmul([tuple(map(int, c.split("x"))) for c in spec.split(",")])
    elif cmd == "fp8attn":
        ns = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1024,2048,4096,8192,16384").split(",")]
        fp8attn(ns, len(sys.argv) > 3 and sys.argv[3] == "1")
    elif cmd == "fp8mm":
        spec = sys.argv[2] if len(sys.argv) > 2 else "1024x1024x1024,2048x2048x2048,4096x4096x4096,8192x8192x8192,16384x2048x8192,16384x8192x2048"
        fp8mm([tuple(map(int, c.split("x"))) for c in spec.split(",")])
    else:
        sys.exit(__doc__)
