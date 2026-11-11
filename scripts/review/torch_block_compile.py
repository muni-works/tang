# PyTorch baselines for the d=576 block (tang paper table "Transformer block"): eager,
# torch.compile, and CUDA-graphed (torch.cuda.graph), fp32 and TF32, forward and
# forward+backward, seq 128 and 512. Same block as scripts/torch_bench.py.
# Prints one line per configuration; a configuration that fails prints the error.
import sys, torch, torch.nn.functional as F
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
D, H, KV, HD, FF = 576, 9, 3, 64, 1536
def bench(fn, iters=50, warm=10, batches=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(batches):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best
class Block(torch.nn.Module):
    def __init__(s):
        super().__init__()
        lin = lambda i, o: torch.nn.Linear(i, o, bias=True)
        s.wq, s.wk, s.wv, s.wo = lin(D, D), lin(D, KV*HD), lin(D, KV*HD), lin(D, D)
        s.gate, s.up, s.down = lin(D, FF), lin(D, FF), lin(FF, D)
    def rms(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    def rope(s, x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
    def forward(s, x, cos, sin):
        T = x.shape[0]
        h = s.rms(x)
        q = s.rope(s.wq(h).view(T,H,HD), cos, sin); k = s.rope(s.wk(h).view(T,KV,HD), cos, sin); v = s.wv(h).view(T,KV,HD)
        q, k, v = (t.transpose(0,1).unsqueeze(0) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + s.wo(a.squeeze(0).transpose(0,1).reshape(T, D))
        h = s.rms(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
for T in (128, 512):
    x = torch.randn(T, D, device=dev); target = torch.randn(T, D, device=dev)
    pos = torch.arange(T, device=dev).float().unsqueeze(1); i = torch.arange(HD//2, device=dev).float()
    theta = pos / (10000 ** (2*i/HD)); cos, sin = theta.cos().unsqueeze(1), theta.sin().unsqueeze(1)
    blk = Block().to(dev)
    def one(mode, prec, bwd):
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        m = blk
        if mode == "compile":
            m = torch.compile(blk)
        def fwd():
            with torch.no_grad():
                return m(x, cos, sin)
        def step():
            for p in blk.parameters():
                p.grad = None if mode != "graph" else p.grad
            if mode == "graph":
                blk.zero_grad(set_to_none=False)
            F.mse_loss(m(x, cos, sin), target).backward()
        fn = step if bwd else fwd
        if mode == "graph":
            if bwd:
                for p in blk.parameters():
                    p.grad = torch.zeros_like(p)
            g = torch.cuda.CUDAGraph()
            for _ in range(3): fn()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn()
            fn = g.replay
        return bench(fn)
    for mode in ("eager", "compile", "graph"):
        for prec in ("fp32", "tf32"):
            for bwd in (False, True):
                name = f"block {'fwd+bwd' if bwd else 'fwd'} seq={T} {prec} {mode}"
                try:
                    print(f"{name}: {one(mode, prec, bwd):.3f} ms", flush=True)
                except Exception as e:
                    print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
