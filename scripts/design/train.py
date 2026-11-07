"""Small-LM training for the design-space study (docs/design.md).

One GPT-style model per run; only the sequence mixer changes. Mixers:

  softmax         causal softmax attention (SDPA / FlashAttention)        baseline
  mamba2          Mamba-2 SSD (fla)                                        baseline
  gdn             gated delta net (fla)                                    baseline
  forget          s + c_i - c_j, softmax  (c = cumsum log sigmoid gate)    tang: lifts (fwd, dq dk dv)
  logdist         s - log(1 + i - j), softmax                               tang: lifts
  sigmoid         sigmoid(s - log T), unnormalized                          tang: lifts
  forget_sigmoid  sigmoid(s + c_i - c_j - log T)                            tang: lifts (fwd, dq dk dv)
  softcap_forget  30 tanh(s/30) + c_i - c_j, softmax                        tang: lifts (fwd, dq dk dv)

usage: python train.py MIXER --data DIR --tokens 1e9 [--out res.jsonl]
Prints one JSON line per eval: step, tokens, wall seconds, train/val loss.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("mixer")
p.add_argument("--data", default="/data")
p.add_argument("--tokens", type=float, default=1e9)
p.add_argument("--layers", type=int, default=12)
p.add_argument("--dim", type=int, default=768)
p.add_argument("--heads", type=int, default=12)
p.add_argument("--seq", type=int, default=1024)
p.add_argument("--micro", type=int, default=32)
p.add_argument("--batch", type=int, default=64)  # sequences per step
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--out", default=None)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
T, D, H = a.seq, a.dim, a.heads
HD = D // H


def rope(x):  # x [B, H, T, hd]
    hd = x.shape[-1]
    f = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    t = torch.arange(x.shape[-2], device=x.device).float()
    ang = torch.outer(t, f)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


_flex = None


def flex(q, k, v, score_mod):
    global _flex
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if _flex is None:
        _flex = (torch.compile(flex_attention, dynamic=False),
                 create_block_mask(lambda b, h, i, j: j <= i, None, None, T, T, device=dev))
    fn, bm = _flex
    return fn(q, k, v, score_mod=score_mod, block_mask=bm)


class Attn(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.gated = "forget" in kind
        if self.gated:
            self.fg = nn.Linear(D, H, bias=True)
            nn.init.constant_(self.fg.bias, 3.0)  # start near "no forgetting"

    def forward(self, x):
        B = x.shape[0]
        q, k, v = self.qkv(x).view(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        kind = self.kind
        if self.gated:
            c = F.logsigmoid(self.fg(x).float()).transpose(1, 2).cumsum(-1)  # [B, H, T]
        if kind == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif kind in ("forget", "softcap_forget", "logdist"):
            if kind == "logdist":
                mod = lambda s, b, h, i, j: s - torch.log1p((i - j).clamp(min=0).float())
            elif kind == "forget":
                mod = lambda s, b, h, i, j: s + (c[b, h, i] - c[b, h, j])
            else:
                mod = lambda s, b, h, i, j: 30 * torch.tanh(s / 30) + (c[b, h, i] - c[b, h, j])
            y = flex(q, k, v, mod)
        else:  # sigmoid family: no flash kernel in torch; explicit masked form
            s = (q @ k.transpose(-1, -2)).float() * HD ** -0.5 - math.log(T)
            if self.gated:
                s = s + c[..., :, None] - c[..., None, :]
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            w = torch.sigmoid(s).masked_fill(~mask, 0).to(v.dtype)
            y = w @ v
        return self.o(y.transpose(1, 2).reshape(B, T, D))


def mixer(kind):
    if kind == "mamba2":
        from fla.layers import Mamba2
        return Mamba2(hidden_size=D, num_heads=2 * D // 64, head_dim=64, state_size=128,
                      expand=2, n_groups=1, chunk_size=64, layer_idx=0)
    if kind == "gdn":
        from fla.layers import GatedDeltaNet
        return GatedDeltaNet(hidden_size=D, head_dim=128, num_heads=D // 128 * 1, expand_v=2,
                             mode="chunk", use_short_conv=True, layer_idx=0)
    return Attn(kind)


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.mix = mixer(kind)
        self.up = nn.Linear(D, 4 * D, bias=False)
        self.down = nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        m = self.mix(self.n1(x))
        if isinstance(m, tuple):
            m = m[0]
        x = x + m
        return x + self.down(F.gelu(self.up(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.emb = nn.Embedding(50304, D)
        self.blocks = nn.ModuleList(Block(kind) for _ in range(a.layers))
        self.norm = nn.RMSNorm(D)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear) and m.weight.dim() == 2 and not hasattr(m, "_fla"):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, tgt):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.norm(x), self.emb.weight).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")


def batch(src, n, gen):
    ix = gen.integers(0, len(src) - T - 1, n)
    x = torch.from_numpy(np.stack([src[i:i + T + 1].astype(np.int64) for i in ix]))
    x = x.pin_memory().to(dev, non_blocking=True)
    return x[:, :-1], x[:, 1:]


model = GPT(a.mixer).to(dev)
nparam = sum(p.numel() for p in model.parameters())
nmix = sum(p.numel() for b in model.blocks for p in b.mix.parameters())
cmodel = model if a.mixer in ("mamba2", "gdn") else torch.compile(model)
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
steps = int(a.tokens // (a.batch * T))
warm = min(500, steps // 20)
accum = a.batch // a.micro
gen = np.random.default_rng(a.seed)
vgen_seed = 1234
out = open(a.out, "a") if a.out else None


def log(d):
    d.update(mixer=a.mixer, params=nparam, mixer_params=nmix)
    s = json.dumps(d)
    print("RES " + s, flush=True)
    if out:
        out.write(s + "\n")
        out.flush()


@torch.no_grad()
def evaluate():
    model.eval()
    g = np.random.default_rng(vgen_seed)
    tot = 0.0
    nb = 40
    for _ in range(nb):
        x, y = batch(val, a.micro, g)
        with torch.autocast("cuda", torch.bfloat16):
            tot += cmodel(x, y).item()
    model.train()
    return tot / nb


t0 = time.time()
tl = 0.0
for step in range(steps + 1):
    if step % a.eval_every == 0 or step == steps:
        torch.cuda.synchronize()
        log(dict(step=step, tokens=step * a.batch * T, wall=time.time() - t0,
                 train=tl, val=evaluate()))
    if step == steps:
        break
    lr = a.lr * min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / steps))))
    for gp in opt.param_groups:
        gp["lr"] = lr
    tl = 0.0
    for _ in range(accum):
        x, y = batch(train, a.micro, gen)
        with torch.autocast("cuda", torch.bfloat16):
            loss = cmodel(x, y) / accum
        loss.backward()
        tl += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step == 50:
        torch.cuda.synchronize()
        t50 = time.time()
    if step == 250:
        torch.cuda.synchronize()
        log(dict(throughput=200 * a.batch * T / (time.time() - t50), step=step))
