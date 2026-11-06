"""PyTorch reference for `cargo run --release -p tang-bench --example lowering_caps`:
the same matmul, attention forward (SDPA) and attention forward+backward
shapes, on MPS or CUDA. Kernel-only style: 30 calls per sample, one sync,
best of 5; f32 and f16.

    python scripts/torch_lowering_caps.py
"""
import time

import torch
import torch.nn.functional as F

dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
sync = torch.cuda.synchronize if dev.type == "cuda" else torch.mps.synchronize
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
ITERS, ROUNDS = 30, 5


def best(f):
    for _ in range(3):
        f()
    sync()
    b = float("inf")
    for _ in range(ROUNDS):
        t = time.perf_counter()
        for _ in range(ITERS):
            f()
        sync()
        b = min(b, (time.perf_counter() - t) / ITERS * 1e3)
    return b


print(f"torch {torch.__version__} on {dev} ({torch.cuda.get_device_name() if dev.type == 'cuda' else 'MPS'}), ms; f32 matmul without TF32")
for m, k, n in [(1024, 1024, 1024), (512, 576, 1536), (2048, 2048, 2048)]:
    row = []
    for dt in (torch.float32, torch.float16):
        a = torch.rand(m, k, device=dev, dtype=dt) * 2 - 1
        b = torch.rand(k, n, device=dev, dtype=dt) * 2 - 1
        row.append(best(lambda: a @ b))
    print(f"matmul {m}x{k}x{n}: f32 {row[0]:.3f}  f16 {row[1]:.3f}")
d = 64
for h, n, causal in [(1, 1024, False), (8, 1024, False), (8, 2048, False), (8, 2048, True), (8, 4096, False)]:
    row = []
    for dt in (torch.float32, torch.float16):
        q, k, v = (torch.rand(1, h, n, d, device=dev, dtype=dt) * 2 - 1 for _ in range(3))
        row.append(best(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)))
    print(f"attn fwd h{h} n{n} causal={causal}: sdpa f32 {row[0]:.3f}  f16 {row[1]:.3f}")
for h, n in [(1, 512), (8, 1024)]:
    row = []
    for dt in (torch.float32, torch.float16):
        q, k, v = (torch.rand(1, h, n, d, device=dev, dtype=dt, requires_grad=True) for _ in range(3))
        dy = torch.rand(1, h, n, d, device=dev, dtype=dt)

        def step():
            y = F.scaled_dot_product_attention(q, k, v)
            torch.autograd.grad(y, (q, k, v), dy)

        row.append(best(step))
    print(f"attn fwd+bwd h{h} n{n}: sdpa f32 {row[0]:.3f}  f16 {row[1]:.3f}")
