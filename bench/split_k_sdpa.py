"""PyTorch reference for `cargo run --example split_k`: batch-1 decode
attention (8 kv heads x 4 query heads, head dim 128) via SDPA, and the
long-K matmul [16, K] @ [K, 256], f32 (and bf16 for SDPA). CUDA-event
timed, min over 5 batches of 50."""
import sys
import torch
import torch.nn.functional as F


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best * 1e3  # us


def main():
    hk, gq, d = 8, 4, 128
    for n in [1024, 4096, 16384, 32768]:
        for dt in [torch.float32, torch.bfloat16]:
            q = torch.randn(1, hk * gq, 1, d, device="cuda", dtype=dt)
            k = torch.randn(1, hk, n, d, device="cuda", dtype=dt)
            v = torch.randn(1, hk, n, d, device="cuda", dtype=dt)
            us = bench(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
            gb = k.element_size() * 2 * hk * n * d / (us * 1e-6) / 1e9
            print(f"sdpa {str(dt):15} kv={n:<6} {us:8.1f} us {gb:6.0f} GB/s")
    for kk in [4096, 32768, 131072]:
        a = torch.randn(16, kk, device="cuda")
        b = torch.randn(kk, 256, device="cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        us = bench(lambda: a @ b)
        print(f"torch matmul f32 16x{kk}x256 {us:8.1f} us")


if __name__ == "__main__":
    sys.exit(main())
