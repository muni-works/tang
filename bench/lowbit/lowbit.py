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


def quantize(w, fmt, seed):
    kind, p, _ = FORMATS[fmt]
    w = w.float()
    had = p.get("had", False)
    if had:
        w = rotate(w, seed)
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
    q = quantize(bf16_tensor(name), fmt, seed=zlib.crc32(name.encode()) & 0xFFFF)
    if FORMATS[fmt][0] == "vq":
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


def nblocks(m):
    return m.config.num_hidden_layers


def proj_params(m, layer=0):
    return sum(m.get_submodule(f"model.layers.{layer}.{p}").weight.numel() for p in PROJ)


def main():
    from transformers import AutoTokenizer
    cmd = sys.argv[1]
    if cmd == "texts":
        return texts()
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
