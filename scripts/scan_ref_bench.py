"""Reference timings for the chunked scan benchmark (crates/tang-bench/examples/scan_bench.rs).

    python scripts/scan_ref_bench.py MODE H T [N M] [--bwd]

MODE: gdn  -> fla.ops.gated_delta_rule.chunk_gated_delta_rule
      ssd  -> mamba_ssm mamba_chunk_scan_combined (headdim M, dstate N, one group per head)
              falling back to fla chunk_simple_gla (the same recurrence, scalar gate)
      lin  -> fla chunk_simple_gla with g = 0 (plain linear attention)
Inputs bf16, batch 1, the shapes tang uses. Prints the median of 20 runs (ms),
forward, or forward + backward with --bwd.
"""
import sys

import torch
import torch.nn.functional as F

mode, H, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rest = [a for a in sys.argv[4:] if not a.startswith("--")]
N = int(rest[0]) if rest else 128
M = int(rest[1]) if len(rest) > 1 else 128
bwd = "--bwd" in sys.argv
dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[10]


def leaf(*s, f=None):
    x = torch.randn(*s, device=dev, dtype=torch.float32)
    if f is not None:
        x = f(x)
    return x.to(dt).requires_grad_(bwd)


name = mode
if mode == "gdn":
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q = leaf(1, T, H, N, f=lambda x: F.normalize(x, dim=-1))
    k = leaf(1, T, H, N, f=lambda x: F.normalize(x, dim=-1))
    v = leaf(1, T, H, M)
    g = (-0.5 * torch.rand(1, T, H, device=dev)).requires_grad_(bwd)
    beta = torch.rand(1, T, H, device=dev).to(dt).requires_grad_(bwd)

    def run():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=N ** -0.5)
        if bwd:
            o.sum().backward()

    name = "fla chunk_gated_delta_rule"
elif mode == "ssd":
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        x = leaf(1, T, H, M)
        dtv = (0.001 + 0.099 * torch.rand(1, T, H, device=dev)).requires_grad_(bwd)
        A = (-torch.ones(H, device=dev)).requires_grad_(bwd)
        B = leaf(1, T, H, N)
        C = leaf(1, T, H, N)

        def run():
            y = mamba_chunk_scan_combined(x, dtv, A, B, C, chunk_size=64)
            if bwd:
                y.sum().backward()

        name = "mamba_ssm mamba_chunk_scan_combined"
    except ImportError:
        from fla.ops.simple_gla import chunk_simple_gla

        q, k, v = leaf(1, T, H, N), leaf(1, T, H, N), leaf(1, T, H, M)
        g = (-0.1 * torch.rand(1, T, H, device=dev)).requires_grad_(bwd)

        def run():
            o, _ = chunk_simple_gla(q, k, v, g, scale=1.0)
            if bwd:
                o.sum().backward()

        name = "fla chunk_simple_gla (mamba_ssm missing)"
else:
    from fla.ops.simple_gla import chunk_simple_gla

    q = leaf(1, T, H, N, f=lambda x: F.normalize(x, dim=-1))
    k = leaf(1, T, H, N, f=lambda x: F.normalize(x, dim=-1))
    v = leaf(1, T, H, M)
    g = torch.zeros(1, T, H, device=dev)

    def run():
        o, _ = chunk_simple_gla(q, k, v, g, scale=N ** -0.5)
        if bwd:
            o.sum().backward()

    name = "fla chunk_simple_gla g=0"

ms = timeit(run)
print(f"{mode} H={H} T={T} n={N} m={M} {'fwd+bwd' if bwd else 'fwd'} {ms:.3f} ms  [{name}, {torch.cuda.get_device_name()}]")
