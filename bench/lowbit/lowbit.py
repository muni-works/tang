#!/usr/bin/env python3
"""Low-bit weight spike for Qwen3-8B on CPU (see bench/results/lowbit/README.md).

Fake-quantizes the decoder-block projections of a bf16 Qwen3-8B with
scalar (affine, per-group) or vector (k-means codebook) quantizers,
optionally after a block-Hadamard incoherence rotation, and measures
perplexity on a small code+prose set and greedy-token agreement on coding
prompts. The 4-bit baseline is mlx-community/Qwen3-8B-4bit's own
weights, dequantized. Reference algorithms (with tests) live in
crates/tang-compile/src/lowbit.rs; this file is their torch twin, for speed.

    python3 lowbit.py texts            # build the eval set (run from repo root)
    python3 lowbit.py fmt <name>...    # whole-model eval of formats
    python3 lowbit.py sens <name>      # per-block sensitivity
    python3 lowbit.py plan <json>      # eval a per-block plan {block: fmt}
"""
import json, math, os, sys, time, glob, zlib
import torch

BF16 = os.path.expanduser("~/models/Qwen3-8B")
MLX = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/*"))
OUT = os.path.expanduser("~/lowbit/out")
CACHE = os.path.expanduser("~/lowbit/cache")
PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
torch.set_num_threads(int(os.environ.get("THREADS", "12")))

# name -> (kind, params, bits/weight including scales/codebook amortization)
FORMATS = {
    "a4": ("mlx", {}, 4.5),
    "a3g64": ("affine", dict(bits=3, g=64), 3.5),
    "a3g128": ("affine", dict(bits=3, g=128), 3.25),
    "a2g64": ("affine", dict(bits=2, g=64), 2.5),
    "a3g64h": ("affine", dict(bits=3, g=64, had=True), 3.5),
    "a2g64h": ("affine", dict(bits=2, g=64, had=True), 2.5),
    "vq2d64": ("vq", dict(d=2, k=64, g=128), 3.125),
    "vq4d2048": ("vq", dict(d=4, k=2048, g=128), 2.875),
    "vq4d256": ("vq", dict(d=4, k=256, g=128), 2.125),
    "vq4d2048h": ("vq", dict(d=4, k=2048, g=128, had=True), 2.875),
    "vq4d256h": ("vq", dict(d=4, k=256, g=128, had=True), 2.125),
    "a4g128h": ("affine", dict(bits=4, g=128, had=True), 4.25),
    "vq2d128h": ("vq", dict(d=2, k=128, g=128, had=True), 3.625),
    "vq2d256h": ("vq", dict(d=2, k=256, g=128, had=True), 4.125),
    "vq4d4096h": ("vq", dict(d=4, k=4096, g=128, had=True), 3.125),
    "vq4d8192h": ("vq", dict(d=4, k=8192, g=128, had=True), 3.375),
    # GPTQ (Hessian-aware, error feedback; `calib` builds the Hessians)
    "g3g64h": ("affine", dict(bits=3, g=64, had=True, gptq=True), 3.5),
    "g3g64": ("affine", dict(bits=3, g=64, gptq=True), 3.5),
    # rotation seeded per projection *input* (q/k/v share one, gate/up one):
    # what a graph can apply once to the activation
    "g3g64hs": ("affine", dict(bits=3, g=64, had=True, shared=True, gptq=True), 3.5),
    "g3g128h": ("affine", dict(bits=3, g=128, had=True, gptq=True), 3.25),
    "g4g128h": ("affine", dict(bits=4, g=128, had=True, gptq=True), 4.25),
    "gvq4d2048h": ("vq", dict(d=4, k=2048, g=128, had=True, gptq=True), 2.875),
    "gvq4d4096h": ("vq", dict(d=4, k=4096, g=128, had=True, gptq=True), 3.125),
    "gvq4d8192h": ("vq", dict(d=4, k=8192, g=128, had=True, gptq=True), 3.375),
    "vq8d4096h": ("vq", dict(d=8, k=4096, g=128, had=True), 1.625),
}


# ---------------------------------------------------------------- quantizers
def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


HB = 128  # Hadamard block along the input dim (divides 4096 and 12288)


def rotate(w, seed, inverse=False):
    """Randomized block Hadamard on the input dim: W' = (W * s) H per block.
    At inference the same rotation is applied to x (x' = (x * s) H), so
    W' x' = W x; here we just undo it after dequantizing."""
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (w.shape[1],), generator=g) * 2 - 1).float()
    h = hadamard(HB)
    o, i = w.shape
    if not inverse:
        return ((w * s).reshape(o, i // HB, HB) @ h).reshape(o, i)
    return ((w.reshape(o, i // HB, HB) @ h.T).reshape(o, i)) * s


def affine(w, bits, g):
    o, i = w.shape
    x = w.reshape(o, i // g, g)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    sc = ((hi - lo) / qmax).clamp_min(1e-8)
    sc, lo = sc.half().float(), lo.half().float()  # stored as f16, like MLX
    q = ((x - lo) / sc).round().clamp(0, qmax)
    return (q * sc + lo).reshape(o, i)


def kmeans(v, k, iters=12, sample=1 << 18, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = v.shape[0]
    fit = v[torch.randperm(n, generator=g)[:min(n, max(sample, 32 * k))]]
    c = fit[torch.randperm(fit.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = assign(fit, c)
        s = torch.zeros_like(c).index_add_(0, a, fit)
        cnt = torch.bincount(a, minlength=k).float().unsqueeze(1)
        dead = cnt.squeeze(1) == 0
        c = torch.where(dead.unsqueeze(1), c, s / cnt.clamp_min(1))
        if dead.any():  # reseed empty clusters from random points
            c[dead] = fit[torch.randint(0, fit.shape[0], (int(dead.sum()),), generator=g)]
    return c


def assign(v, c, chunk=1 << 16):
    cn = (c * c).sum(1)
    out = torch.empty(v.shape[0], dtype=torch.long)
    for s in range(0, v.shape[0], chunk):
        x = v[s:s + chunk]
        out[s:s + chunk] = (cn - 2 * x @ c.T).argmin(1)
    return out


def vq(w, d, k, g):
    """Per-group RMS scale (f16) along the input dim, then one k-means
    codebook (k entries of d dims) per matrix over the normalized
    d-vectors. Stored: log2(k)/d bits per weight + 16/g + codebook."""
    o, i = w.shape
    x = w.reshape(o, i // g, g)
    sc = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8).half().float()
    v = (x / sc).reshape(-1, d)
    c = kmeans(v, k)
    return (c[assign(v, c)].reshape(o, i // g, g) * sc).reshape(o, i)


HESS = os.path.expanduser("~/lowbit/hess")
# the projection whose input each projection sees (q/k/v and gate/up share one)
HIN = {"self_attn.q_proj": "q", "self_attn.k_proj": "q", "self_attn.v_proj": "q",
       "self_attn.o_proj": "o", "mlp.gate_proj": "gate", "mlp.up_proj": "gate", "mlp.down_proj": "down"}


def hessian(name, seed, had):
    """E[x x^T] of the projection's input over the calibration set, rotated
    like the weight (R^T H R, R = D Hblk), so GPTQ runs in the stored basis."""
    layer, proj = name.split(".")[2], ".".join(name.split(".")[3:5])
    h = torch.load(f"{HESS}/{layer}.{HIN[proj]}.pt").double()
    if had:
        h = rotate(rotate(h.float(), seed).T.contiguous(), seed).T.double()
    return h


def gptq_inv(h, damp=0.01):
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    h += damp * torch.diag(h).mean() * torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    l = torch.linalg.cholesky(h)
    return torch.linalg.cholesky(torch.cholesky_inverse(l), upper=True).float()


def gptq_affine(w, h, bits, g, blk=128, codes=False):
    """GPTQ (Frantar et al.) with per-group min/max affine params, fixed
    when the sweep enters each group (from the error-updated weights).
    codes=True: (integer codes, f16 scales, f16 biases) instead of the
    dequantized weights. Runs on the GPU when there is one (same math)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.clone().to(dev)
    u = gptq_inv(h.to(dev))
    q = torch.zeros_like(w)
    o, n = w.shape
    if codes:
        qi = torch.zeros(o, n, dtype=torch.uint8, device=dev)
        scs = torch.zeros(o, n // g, dtype=torch.float16, device=dev)
        los = torch.zeros(o, n // g, dtype=torch.float16, device=dev)
    qmax = 2 ** bits - 1
    for b0 in range(0, n, blk):
        b1 = min(b0 + blk, n)
        wb, eb = w[:, b0:b1].clone(), torch.zeros(o, b1 - b0, device=dev)
        for j in range(b1 - b0):
            c = b0 + j
            if c % g == 0:
                grp = torch.cat([wb[:, j:], w[:, b1:]], 1)[:, :g]
                lo, hi = grp.amin(1), grp.amax(1)
                sc = ((hi - lo) / qmax).clamp_min(1e-8).half().float()
                lo = lo.half().float()
                if codes:
                    scs[:, c // g], los[:, c // g] = sc.half(), lo.half()
            x = wb[:, j]
            ci = ((x - lo) / sc).round().clamp(0, qmax)
            qc = ci * sc + lo
            q[:, c] = qc
            if codes:
                qi[:, c] = ci.to(torch.uint8)
            e = (x - qc) / u[c, c]
            wb[:, j:] -= e.unsqueeze(1) * u[c, c:b1].unsqueeze(0)
            eb[:, j] = e
        w[:, b1:] -= eb @ u[b0:b1, b1:]
    return (qi.cpu(), scs.cpu(), los.cpu()) if codes else q.cpu()


def pack_bits(q, bits):
    """Codes (uint8, row-major) as one little-endian bit stream in u32 words,
    code i at bit bits·i (tang's Packing::U3 for bits=3): int32 tensor."""
    import numpy as np
    v = q.reshape(-1).numpy().astype(np.uint64)
    assert v.size * bits % 32 == 0
    out = np.zeros(v.size * bits // 32 + 1, dtype=np.uint64)
    pos = np.arange(v.size, dtype=np.uint64) * bits
    np.bitwise_or.at(out, (pos >> 5).astype(np.int64), (v << (pos & 31)) & 0xFFFFFFFF)
    np.bitwise_or.at(out, (pos >> 5).astype(np.int64) + 1, v << (pos & 31) >> 32)
    return torch.from_numpy(out[:-1].astype(np.uint32).view(np.int32))


def export(fmt, outdir):
    """The MLX 4-bit model with every decoder-block projection replaced by
    GPTQ codes of `fmt` (affine, no rotation): `.weight` the bit stream as
    U32 [rows, cols·bits/32], `.scales` / `.biases` F16 [rows, cols/g].
    Everything else (embeddings, LM head, norms) is MLX's. tang reads it
    with TANG_QWEN_G3=1."""
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    kind, p, _ = FORMATS[fmt]
    assert kind == "affine" and p.get("gptq"), "export: GPTQ affine"
    assert not p.get("had") or p.get("shared"), "export: a rotation must be per input (shared)"
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(MLX[0]):
        if f != "model.safetensors" and not f.startswith("."):
            shutil.copyfile(f"{MLX[0]}/{f}", f"{outdir}/{f}")  # no chmod: network volumes refuse it
    cfg = json.load(open(f"{outdir}/config.json"))
    cfg["quantization"] = dict(cfg.get("quantization", {}), tang_lowbit=fmt, tang_bits=p["bits"],
                               tang_hadamard=HB if p.get("had") else 0)
    json.dump(cfg, open(f"{outdir}/config.json", "w"), indent=2)
    out = {}
    with safe_open(f"{MLX[0]}/model.safetensors", "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    t = time.time()
    for k in list(out):
        base = k[:-len(".weight")] if k.endswith(".weight") else None
        if not base or not any(base.endswith(pr) for pr in PROJ) or ".layers." not in base:
            continue
        name = f"{base}.weight"
        w = bf16_tensor(name).float()
        had = p.get("had", False)
        seed = seed_of(name, fmt)
        if had:
            # codes in the rotated basis; the graph rotates the activation
            # by the same signs (`.had` [cols], ±1 as f16) and block Hadamard
            w = rotate(w, seed)
            out[f"{base}.had"] = signs_of(w.shape[1], seed).half()
        qi, sc, lo = gptq_affine(w, hessian(name, seed, had), p["bits"], p["g"], codes=True)
        out[name] = pack_bits(qi, p["bits"]).reshape(w.shape[0], -1)
        out[f"{base}.scales"], out[f"{base}.biases"] = sc.contiguous(), lo.contiguous()
        print("export", name, round(time.time() - t), flush=True)
    save_file(out, f"{outdir}/model.safetensors", metadata={"format": "mlx"})
    print("saved", outdir, flush=True)


def gptq_vq(w, h, d, k, g, blk=128):
    """GPTQ-VQ: group RMS scales and the k-means codebook come from the
    plain fit; the sweep then assigns one d-vector at a time against the
    error-updated weights and feeds each column's residual forward."""
    o, n = w.shape
    x = w.reshape(o, n // g, g)
    sc = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8).half().float()
    book = kmeans((x / sc).reshape(-1, d), k)
    sc = sc.reshape(o, n // g)
    w = w.clone()
    u = gptq_inv(h)
    q = torch.zeros_like(w)
    for b0 in range(0, n, blk):
        b1 = min(b0 + blk, n)
        wb, eb = w[:, b0:b1].clone(), torch.zeros(o, b1 - b0)
        for j in range(0, b1 - b0, d):
            c = b0 + j
            s = sc[:, c // g].unsqueeze(1)
            qv = book[assign(wb[:, j:j + d] / s, book)] * s
            for t in range(d):
                q[:, c + t] = qv[:, t]
                e = (wb[:, j + t] - qv[:, t]) / u[c + t, c + t]
                wb[:, j + t:] -= e.unsqueeze(1) * u[c + t, c + t:b1].unsqueeze(0)
                eb[:, j + t] = e
        w[:, b1:] -= eb @ u[b0:b1, b1:]
    return q


def quantize(w, fmt, seed, name=None):
    kind, p, _ = FORMATS[fmt]
    w = w.float()
    had = p.get("had", False)
    if had:
        w = rotate(w, seed)
    if p.get("gptq"):
        h = hessian(name, seed, had)
        q = gptq_affine(w, h, p["bits"], p["g"]) if kind == "affine" else gptq_vq(w, h, p["d"], p["k"], p["g"])
    else:
        q = affine(w, p["bits"], p["g"]) if kind == "affine" else vq(w, p["d"], p["k"], p["g"])
    if had:
        q = rotate(q, seed, inverse=True)
    return q.bfloat16()


# ------------------------------------------------------------------- weights
def bf16_tensor(name):
    from safetensors import safe_open
    idx = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{BF16}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


def mlx_tensor(f, k):
    """One tensor of mlx-community/Qwen3-8B-4bit, dequantized to bf16 (affine
    4-bit, group 64, f16 scale+bias, codes packed 8 per u32, low bits first)."""
    base = k[:-len(".weight")]
    if f"{base}.scales" not in f.keys():
        return f.get_tensor(k).bfloat16()
    p = f.get_tensor(k).to(torch.int64) & 0xFFFFFFFF
    q = ((p.unsqueeze(-1) >> torch.arange(0, 32, 4)) & 15).reshape(p.shape[0], -1).float()
    sc, bi = f.get_tensor(f"{base}.scales").float(), f.get_tensor(f"{base}.biases").float()
    g = q.shape[1] // sc.shape[1]
    w = q.reshape(q.shape[0], -1, g) * sc.unsqueeze(-1) + bi.unsqueeze(-1)
    return w.reshape(q.shape[0], -1).bfloat16()


def mlx_file():
    from safetensors import safe_open
    return safe_open(f"{MLX[0]}/model.safetensors", "pt")


def seed_of(name, fmt):
    """The rotation seed of a projection: per matrix, or (`shared`) per
    projection input — `layer.q` for q/k/v, `layer.gate` for gate/up."""
    if FORMATS[fmt][1].get("shared"):
        layer, proj = name.split(".")[2], ".".join(name.split(".")[3:5])
        name = f"{layer}.{HIN[proj]}"
    return zlib.crc32(name.encode()) & 0xFFFF


def signs_of(cols, seed):
    """The ±1 vector `rotate` multiplies by (same generator)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (cols,), generator=g) * 2 - 1).float()


def cached(name, fmt, layer):
    """Quantized weight for one projection. VQ results (minutes per matrix on
    CPU) are cached on disk as bf16; the rest is recomputed."""
    if fmt == "bf16":
        return bf16_tensor(name)
    if fmt == "a4":
        return mlx_tensor(mlx_file(), name)
    path = f"{CACHE}/{fmt}/{name}.pt"
    if os.path.exists(path):
        return torch.load(path)
    q = quantize(bf16_tensor(name), fmt, seed=seed_of(name, fmt), name=name)
    if FORMATS[fmt][0] == "vq" or FORMATS[fmt][1].get("gptq"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(q, path)
    return q


def load_model(base="a4"):
    """The model with every tensor from `base` ("a4": the MLX 4-bit weights,
    including embeddings and lm_head; "bf16": the original)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(BF16, torch_dtype=torch.bfloat16)
    m.eval()
    if base == "a4":
        f = mlx_file()
        sd = m.state_dict()
        for k in f.keys():
            if k.endswith(".weight") and k in sd:
                sd[k].copy_(mlx_tensor(f, k))
    return m


def set_block(m, layer, fmt):
    for p in PROJ:
        name = f"model.layers.{layer}.{p}.weight"
        mod = m.get_submodule(f"model.layers.{layer}.{p}")
        mod.weight.data.copy_(cached(name, fmt, layer))


# ---------------------------------------------------------------------- eval
def texts():
    root = os.getcwd()
    code = "".join(open(f"{root}/crates/tang-la/src/{f}").read()
                   for f in ["cholesky.rs", "qr.rs", "lu.rs"])
    prose = "".join(open(f"{root}/{f}").read() for f in ["README.md", "NOTES.md", "CONTRIBUTING.md"])
    prompts = [
        "def quicksort(arr):\n",
        "// Rust: parse a CSV line into fields, handling quoted commas\nfn parse_csv_line(line: &str) -> Vec<String> {\n",
        "#include <stdio.h>\n\n/* reverse a singly linked list in place */\nstruct node *reverse(struct node *head) {\n",
        "function debounce(fn, ms) {\n",
        "-- SQL: top 5 customers by total order value in 2024\nSELECT",
        "class LRUCache:\n    def __init__(self, capacity: int):\n",
        "fn binary_search<T: Ord>(xs: &[T], target: &T) -> Option<usize> {\n",
        "import numpy as np\n\ndef softmax(x, axis=-1):\n",
        "package main\n\nimport \"fmt\"\n\n// fizzbuzz prints 1..n\nfunc fizzbuzz(n int) {\n",
        "# bash: find the 10 largest files under the current directory\n",
    ]
    os.makedirs("bench/lowbit", exist_ok=True)
    json.dump(dict(code=code, prose=prose, prompts=prompts), open("bench/lowbit/texts.json", "w"))


def eval_ids(tok, ntok):
    t = json.load(open(os.path.join(os.path.dirname(__file__), "texts.json")))
    c = tok(t["code"])["input_ids"][:ntok // 2]
    p = tok(t["prose"])["input_ids"][:ntok // 2]
    return [torch.tensor([c]), torch.tensor([p])], t["prompts"]


@torch.no_grad()
def ppl(m, seqs, win=1024):
    nll, n = 0.0, 0
    for ids in seqs:
        for s in range(0, ids.shape[1] - 1, win):
            x = ids[:, s:s + win + 1]
            lg = m(x[:, :-1]).logits.float()
            nll += torch.nn.functional.cross_entropy(lg[0], x[0, 1:], reduction="sum").item()
            n += x.shape[1] - 1
    return math.exp(nll / n)


@torch.no_grad()
def greedy_refs(m, tok, prompts, n=32):
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"]
        g = m.generate(ids, max_new_tokens=n, do_sample=False)
        out.append(g[0].tolist() + [ids.shape[1]])  # last item: prompt length
    return out


@torch.no_grad()
def agreement(m, refs):
    hit, tot = 0, 0
    for r in refs:
        pl, ids = r[-1], torch.tensor([r[:-1]])
        pred = m(ids).logits[0, pl - 1:-1].argmax(-1)
        hit += (pred == ids[0, pl:]).sum().item()
        tot += ids.shape[1] - pl
    return hit / tot


@torch.no_grad()
def calib(path, nseq, seqlen, first=0, last=99):
    """Accumulate E[x x^T] at the inputs of q/o/gate/down in every block of
    the bf16 model over `nseq` windows of `seqlen` tokens from `path` (text
    disjoint from the eval set), for blocks [first, last) (the f32
    Hessians of all 36 blocks are ~28 GB: split passes on small hosts). Saved f32 to ~/lowbit/hess/<layer>.<in>.pt."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BF16)
    ids = tok(open(path).read())["input_ids"]
    stride = max(seqlen, (len(ids) - seqlen) // nseq)
    seqs = [ids[i:i + seqlen] for i in range(0, len(ids) - seqlen, stride)][:nseq]
    print("calib", len(seqs), "x", seqlen, "of", len(ids), "tokens", flush=True)
    m = load_model("bf16")
    acc, hooks = {}, []
    for l in range(first, min(last, nblocks(m))):
        for p, key in [("self_attn.q_proj", "q"), ("self_attn.o_proj", "o"),
                       ("mlp.gate_proj", "gate"), ("mlp.down_proj", "down")]:
            def hook(mod, inp, _out, k=f"{l}.{key}"):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                acc[k] = acc.get(k, 0) + x.T @ x
            hooks.append(m.get_submodule(f"model.layers.{l}.{p}").register_forward_hook(hook))
    t = time.time()
    for i, s in enumerate(seqs):
        m(torch.tensor([s]))
        if i % 16 == 0:
            print("seq", i, round(time.time() - t), flush=True)
    os.makedirs(HESS, exist_ok=True)
    n = len(seqs) * seqlen
    for k, v in acc.items():
        torch.save(v / n, f"{HESS}/{k}.pt")
    print("saved", len(acc), "hessians", flush=True)


def nblocks(m):
    return m.config.num_hidden_layers


def proj_params(m, layer=0):
    return sum(m.get_submodule(f"model.layers.{layer}.{p}").weight.numel() for p in PROJ)


def main():
    from transformers import AutoTokenizer
    cmd = sys.argv[1]
    if cmd == "texts":
        return texts()
    if cmd == "calib":  # Hessians for GPTQ: calib <calib.txt> [nseq] [seqlen]
        a = sys.argv + [None] * 6
        return calib(a[2], int(a[3] or 128), int(a[4] or 256), int(a[5] or 0), int(a[6] or 99))
    if cmd == "export":  # export <fmt> <outdir>
        return export(sys.argv[2], sys.argv[3])
    if cmd == "cache":  # fill the VQ cache: cache <fmt> <first> <last>
        for l in range(int(sys.argv[3]), int(sys.argv[4])):
            for p in PROJ:
                cached(f"model.layers.{l}.{p}.weight", sys.argv[2], l)
            print("cached", sys.argv[2], l, flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BF16)
    ntok = int(os.environ.get("NTOK", "4096"))
    seqs, prompts = eval_ids(tok, ntok)
    refpath = f"{OUT}/refs.json"
    log = open(f"{OUT}/{cmd}.jsonl", "a")

    def rec(**kw):
        kw["ntok"] = ntok
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    if cmd == "refs":  # bf16 and a4 reference generations + baseline numbers
        m = load_model("bf16")
        t = time.time()
        rb = greedy_refs(m, tok, prompts)
        rec(fmt="bf16", ppl=ppl(m, seqs), secs=time.time() - t)
        m = load_model("a4")
        ra = greedy_refs(m, tok, prompts)
        json.dump(dict(bf16=rb, a4=ra), open(refpath, "w"))
        rec(fmt="a4", ppl=ppl(m, seqs), agree_bf16=agreement(m, rb), agree_a4=1.0)
        return
    refs = json.load(open(refpath))
    m = load_model("a4")
    L = nblocks(m)
    if cmd == "fmt":
        for fmt in sys.argv[2:]:
            t = time.time()
            for l in range(L):
                set_block(m, l, fmt)
            tq = time.time() - t
            rec(fmt=fmt, bits=FORMATS[fmt][2], ppl=ppl(m, seqs), agree_bf16=agreement(m, refs["bf16"]),
                agree_a4=agreement(m, refs["a4"]), quant_secs=tq, secs=time.time() - t)
            for l in range(L):
                set_block(m, l, "a4")
    elif cmd == "sens":
        fmt = sys.argv[2]
        rec(fmt="a4", block=-1, ppl=ppl(m, seqs))
        for l in range(L):
            set_block(m, l, fmt)
            rec(fmt=fmt, block=l, ppl=ppl(m, seqs))
            set_block(m, l, "a4")
    elif cmd == "plan":
        plan = json.load(open(sys.argv[2]))
        bits, n = 0.0, 0
        for l in range(L):
            f = plan.get(str(l), "a4")
            set_block(m, l, f)
            bits += FORMATS[f][2]
            n += 1
        rec(plan=sys.argv[2], bits=bits / n, ppl=ppl(m, seqs), agree_bf16=agreement(m, refs["bf16"]),
            agree_a4=agreement(m, refs["a4"]))


if __name__ == "__main__":
    main()
