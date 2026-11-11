"""DeepSeek-V4.1-Flash reference forward in plain torch: the truth tang's port is checked against.

A transcription of DeepSeek's `inference/model.py` (HF repo deepseek-ai/DeepSeek-V4.1-Flash),
written from reading it, with its tilelang kernels replaced by torch ops of the same numerics.
No DeepSeek code is imported or run. Deliberate differences:

- every top-k breaks ties by lowest index (torch.topk's tie order is unspecified), so the
  fixture is deterministic and the Rust side can match it exactly;
- batch 1, no tensor parallelism, no vision.

Weights are a flat dict in the checkpoint's own naming and dtypes (see `tensor_specs`):
F8_E4M3 with F8_E8M0 32x32 block scales, fp4 experts packed two to an int8 byte along K (low
nibble first) with an E8M0 scale per row x 32, bf16 / f32 elsewhere.

Activations follow the reference's dtypes: bf16 stream, fp32 inside norms, router, mHC and the
compressor's pooling; every fp8 / fp4 GEMM quantizes its activation per 32 with power-of-two
scales first.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BF16 = torch.bfloat16


# --- quantization and the kernels ------------------------------------------------------------


def pow2_ceil(x):
    """2^ceil(log2(x)) for positive fp32 x (kernel.py's fast_round_scale)."""
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


def fp4_round(v):
    """Nearest e2m1 value, ties to the even code."""
    a = v.abs().clamp(max=FP4_MAX)
    d = (a.unsqueeze(-1) - FP4_GRID).abs()
    best = d.min(-1, keepdim=True).values
    idx = torch.arange(8)
    score = torch.where(d == best, (idx % 2) * 16 + idx, torch.full_like(idx, 1 << 20))
    return torch.copysign(FP4_GRID[score.argmin(-1)], v)


def fp4_codes(v):
    code = (v.abs().unsqueeze(-1) == FP4_GRID).float().argmax(-1)
    return code + 8 * ((v < 0) & (v != 0)).long()


def fp4_unpack(b):
    """[N, K/2] int8 -> [N, K] fp32 values."""
    u = b.view(torch.uint8)
    table = torch.cat([FP4_GRID, -FP4_GRID])
    return torch.stack([table[(u & 0xF).long()], table[(u >> 4).long()]], -1).flatten(-2)


def fp8_qdq(x, block=32):
    """act_quant(..., ue8m0, inplace=True): fp8 quant-dequant per `block`, power-of-two scales."""
    z = x.float().unflatten(-1, (-1, block))
    s = pow2_ceil(z.abs().amax(-1, keepdim=True).clamp(min=1e-4) / FP8_MAX)
    return ((z / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)


def fp4_qdq(x, block=32, e4m3_scale=False):
    """fp4_act_quant(..., inplace=True): e8m0 power-of-two scales, or e4m3 scales for the
    compressed KV (block 16)."""
    z = x.float().unflatten(-1, (-1, block))
    amax = z.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp(min=6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = pow2_ceil(amax.clamp(min=6 * 2**-126) / FP4_MAX)
    return (fp4_round((z / s).clamp(-FP4_MAX, FP4_MAX)) * s).flatten(-2).to(x.dtype)


def dequant_fp8(w, s):
    """F8_E4M3 weight with either 32x32 block scales or per-row x 32 scales -> fp32."""
    w, s = w.float(), s.float()
    if s.shape == (w.shape[0], w.shape[1] // 32):
        return w * s.repeat_interleave(32, 1)
    return w * s.repeat_interleave(32, 0)[: w.shape[0]].repeat_interleave(32, 1)[:, : w.shape[1]]


def dequant_fp4(w, s):
    return fp4_unpack(w) * s.float().repeat_interleave(32, 1)


def topk_stable(x, k):
    """Indices of the k largest along the last dim, ties to the lowest index, in score order."""
    return torch.sort(-x.float(), dim=-1, stable=True).indices[..., :k]


def sparse_attn(q, kv, sink, idxs, scale):
    """q [b,m,h,d], kv [b,n,d], idxs [b,m,k] (-1 = none). Softmax over the gathered keys with a
    per-head sink logit in the denominator; probabilities rounded to bf16 before PV."""
    b, m, h, d = q.shape
    idx = idxs.long()
    valid = idx >= 0
    g = torch.gather(kv.float().unsqueeze(1).expand(b, m, -1, d), 2, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, d))
    g = g * valid.unsqueeze(-1)
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g) * scale
    s = s.masked_fill(~valid.unsqueeze(2), -math.inf)
    mx = s.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(s - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(1, 1, h, 1) - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p.to(BF16).float(), g) / denom
    return o.to(q.dtype)


def sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# --- checkpoint layout ------------------------------------------------------------------------


def tensor_specs(c):
    """(name, shape, kind) for every tensor in the checkpoint. kind: fp8 (32x32 scales), fp8row
    (per-row x 32 scales: the engram table), fp4 (int8-packed, per-row x 32 scales), bf16, f32."""
    D, H, hd, hc = c["dim"], c["n_heads"], c["head_dim"], c["hc_mult"]
    G, ol, ql, I = c["o_groups"], c["o_lora_rank"], c["q_lora_rank"], c["moe_inter_dim"]
    mix = (2 + hc) * hc
    specs = [("embed.weight", (c["vocab_size"], D), "bf16"), ("norm.weight", (D,), "bf16"),
             ("head.weight", (c["vocab_size"], D), "bf16")]

    def block(p, layer_id, n_experts):
        r = c["compress_ratios"][layer_id]
        backbone = layer_id < c["n_layers"]
        s = [
            (f"{p}.attn.attn_sink", (H,), "f32"),
            (f"{p}.attn.wq_a.weight", (ql, D), "fp8"),
            (f"{p}.attn.q_norm.weight", (ql,), "bf16"),
            (f"{p}.attn.wq_b.weight", (H * hd, ql), "fp8"),
            (f"{p}.attn.wkv.weight", (hd, D), "fp8"),
            (f"{p}.attn.kv_norm.weight", (hd,), "bf16"),
            (f"{p}.attn.wo_a.weight", (G * ol, H * hd // G), "fp8"),
            (f"{p}.attn.wo_b.weight", (D, G * ol), "fp8"),
            (f"{p}.attn_norm.weight", (D,), "bf16"),
            (f"{p}.ffn_norm.weight", (D,), "bf16"),
            (f"{p}.ffn.gate.weight", (n_experts, D), "bf16"),
            (f"{p}.ffn.gate.bias", (n_experts,), "f32"),
            (f"{p}.ffn.shared_experts.w1.weight", (I, D), "fp8"),
            (f"{p}.ffn.shared_experts.w2.weight", (D, I), "fp8"),
            (f"{p}.ffn.shared_experts.w3.weight", (I, D), "fp8"),
        ]
        for e in range(n_experts):
            s += [(f"{p}.ffn.experts.{e}.w1.weight", (I, D), "fp4"),
                  (f"{p}.ffn.experts.{e}.w2.weight", (D, I), "fp4"),
                  (f"{p}.ffn.experts.{e}.w3.weight", (I, D), "fp4")]
        for kind in ("attn", "ffn"):
            s += [(f"{p}.hc_{kind}_fn", (mix, hc * D), "f32"), (f"{p}.hc_{kind}_base", (mix,), "f32"),
                  (f"{p}.hc_{kind}_scale", (3,), "f32")]
        if backbone and layer_id in c["kv_source_layers"]:
            s += [(f"{p}.attn.compressor.wkv.weight", (hd, D), "bf16"),
                  (f"{p}.attn.compressor.norm.weight", (hd,), "bf16")]
            if r > 1:
                s += [(f"{p}.attn.compressor.wgate.weight", (hd, D), "bf16")]
        if backbone and layer_id in c["index_source_layers"]:
            s += [(f"{p}.attn.indexer.wq_b.weight", (c["index_n_heads"] * c["index_head_dim"], ql), "fp8"),
                  (f"{p}.attn.indexer.weights_proj.weight", (c["index_n_heads"], D), "bf16")]
            if layer_id in c["kv_source_layers"]:
                s += [(f"{p}.attn.indexer.wk.weight", (c["index_head_dim"], hd), "bf16"),
                      (f"{p}.attn.indexer.k_norm.weight", (c["index_head_dim"],), "bf16")]
        if backbone and layer_id in c["engram_layer_ids"]:
            j = c["engram_layer_ids"].index(layer_id)
            cols = (c["engram_max_ngram_size"] - 1) * c["engram_n_heads"]
            ehd = c["engram_head_dim"]
            s += [(f"{p}.engram.embed.weight", (c["engram_num_embeddings"][j], ehd), "fp8row"),
                  (f"{p}.engram.wkv.weight", (D * (hc + 1), cols * ehd), "fp8"),
                  (f"{p}.engram.q_weight", (hc, D), "bf16"), (f"{p}.engram.k_weight", (hc, D), "bf16")]
        return s

    for i in range(c["n_layers"]):
        specs += block(f"layers.{i}", i, c["n_routed_experts"])
    n_mtp = c["n_mtp_layers"]
    for s_ in range(n_mtp):
        p = f"mtp.{s_}"
        specs += block(p, c["n_layers"] + s_, c["dspark_n_routed_experts"] or c["n_routed_experts"])
        if s_ == 0:
            specs += [(f"{p}.main_proj.weight", (D, D * len(c["dspark_target_layer_ids"])), "fp8"),
                      (f"{p}.main_norm.weight", (D,), "bf16")]
        if s_ == n_mtp - 1:
            rk = c["dspark_markov_rank"]
            specs += [(f"{p}.norm.weight", (D,), "bf16"),
                      (f"{p}.markov_head.embed.weight", (c["vocab_size"], rk), "bf16"),
                      (f"{p}.markov_head.head.weight", (c["vocab_size"], rk), "bf16"),
                      (f"{p}.confidence_head.proj.weight", (1, D + rk), "bf16")]
    return specs


# --- engram hashing ---------------------------------------------------------------------------


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def engram_primes(c):
    """[layer][n-gram size][head] bucket moduli: primes drawn in order from engram_vocab_size,
    never reused across heads, n-gram sizes or layers."""
    seen, out = set(), []
    for _ in c["engram_layer_ids"]:
        per_ngram = []
        for _ in range(c["engram_max_ngram_size"] - 1):
            sizes, cur = [], c["engram_vocab_size"] - 1
            for _ in range(c["engram_n_heads"]):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        out.append(per_ngram)
    return out


class EngramHash:
    """Hash ids of the 2..max_ngram_size-grams ending at each position, per engram layer."""

    def __init__(self, c, token_map, compressed_vocab):
        self.c = c
        self.n = c["engram_max_ngram_size"]
        self.token_map = torch.tensor(token_map)
        self.pad = token_map[c["engram_pad_id"]]
        primes = engram_primes(c)
        self.primes = torch.tensor(primes)  # [L, n-1, heads]
        flat = [[p for g in layer for p in g] for layer in primes]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *f[:-1]]) for f in flat]))
        bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
        rows = []
        for lid in c["engram_layer_ids"]:
            v = np.random.default_rng(10007 * lid).integers(0, bound, size=(self.n,), dtype=np.int64)
            rows.append(torch.tensor(v * 2 + 1))
        self.mult = torch.stack(rows)  # [L, n]
        self.cache = torch.empty(c["max_seq_len"], dtype=torch.int64)

    def __call__(self, ids, start):
        T = ids.numel()
        self.cache[start : start + T] = self.token_map[ids]
        pos = torch.arange(start, start + T)
        toks = []
        for shift in range(self.n):
            src = self.cache[(pos - shift).clamp(min=0)]
            toks.append(torch.where(pos < shift, self.pad, src))
        toks = torch.stack(toks, -1)  # [T, n]
        prod = toks.unsqueeze(1) * self.mult  # [T, L, n]
        rolling, hashes = prod[..., 0], []
        for i in range(1, self.n):
            rolling = rolling ^ prod[..., i]
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets  # [T, L, cols]


# --- the model --------------------------------------------------------------------------------


def yarn_freqs(dim, seqlen, orig, base, factor, beta_fast, beta_slow):
    """RoPE as complex exponentials [seqlen, dim/2], YaRN-corrected when orig > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if orig > 0:
        def corr(rot):
            return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(f), f)


def rope(x, freqs, inverse=False):
    """Rotate the last dim as adjacent pairs; returns x's dtype. x [..., T, (h,) d]."""
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    shape = [1] * z.ndim  # [1, T, d/2] or [1, T, 1, d/2]: broadcast over heads
    shape[1] = f.shape[0]
    shape[-1] = f.shape[-1]
    return torch.view_as_real(z * f.view(shape)).flatten(-2).to(x.dtype)


def rms(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


class Model:
    def __init__(self, c, W, token_map=None, compressed_vocab=None):
        self.c, self.W = c, W
        self.hc = c["hc_mult"]
        self.eps = c["norm_eps"]
        self.n = c["n_layers"]
        self.deq = {}
        S = c["max_seq_len"]
        hd = c["head_dim"]
        self.freqs = []
        for lid in range(self.n + c["n_mtp_layers"]):
            if c["compress_ratios"][lid]:
                f = yarn_freqs(c["rope_head_dim"], S, c["original_seq_len"], c["compress_rope_theta"],
                               c["rope_factor"], c["beta_fast"], c["beta_slow"])
            else:
                f = yarn_freqs(c["rope_head_dim"], S, 0, c["rope_theta"], c["rope_factor"], c["beta_fast"], c["beta_slow"])
            self.freqs.append(f)
        self.win = [torch.zeros(c["window_size"], hd, dtype=BF16) for _ in range(self.n + c["n_mtp_layers"])]
        self.ckv, self.idxk, self.cstate = {}, {}, {}
        for lid in c["kv_source_layers"]:
            r = c["compress_ratios"][lid]
            self.ckv[lid] = torch.zeros(S // r, hd, dtype=BF16)
            self.idxk[lid] = torch.zeros(S // r, c["index_head_dim"], dtype=BF16)
            if r > 1:
                self.cstate[lid] = [torch.zeros(r, hd), torch.full((r, hd), -math.inf)]
        self.shared = {}
        self.hash = EngramHash(c, token_map, compressed_vocab) if c["engram_layer_ids"] else None

    # weights

    def w(self, name):
        """Dequantized fp32 weight, cached."""
        if name not in self.deq:
            t = self.W[name]
            sname = name[: -len("weight")] + "scale"
            if t.dtype == torch.int8:
                self.deq[name] = dequant_fp4(t, self.W[sname])
            elif t.dtype == torch.float8_e4m3fn:
                self.deq[name] = dequant_fp8(t, self.W[sname])
            else:
                self.deq[name] = t.float()
        return self.deq[name]

    def linear(self, x, name):
        """`linear` in model.py: fp8 / fp4 weights quantize the activation per 32 first; bf16
        weights run in bf16."""
        t = self.W[name]
        if t.dtype in (torch.int8, torch.float8_e4m3fn):
            return (fp8_qdq(x).float() @ self.w(name).t()).to(BF16)
        return F.linear(x, t.to(x.dtype))

    # attention

    def attention(self, p, lid, x, sp):
        c, T = self.c, x.size(1)
        H, hd, rd = c["n_heads"], c["head_dim"], c["rope_head_dim"]
        r = c["compress_ratios"][lid]
        backbone = lid < self.n
        fr = self.freqs[lid][sp : sp + T]
        qr = rms(self.linear(x, f"{p}.attn.wq_a.weight"), self.W[f"{p}.attn.q_norm.weight"], self.eps)
        q = self.linear(qr, f"{p}.attn.wq_b.weight").unflatten(-1, (H, hd))
        q = torch.cat([q[..., :-rd], rope(q[..., -rd:], fr)], -1)
        kv, idxs = self.window_kv(p, lid, x, fr, sp)
        if r:
            ckv, cidx = self.compress_kv(p, lid, x, qr, sp, kv.size(1))
            kv = torch.cat([kv, ckv.unsqueeze(0)], 1)
            idxs = torch.cat([idxs, cidx], -1)
        return self.attn_out(p, q, kv, idxs, fr)

    def attn_out(self, p, q, kv, idxs, fr):
        c = self.c
        rd, G, ol = c["rope_head_dim"], c["o_groups"], c["o_lora_rank"]
        o = sparse_attn(q, kv, self.W[f"{p}.attn.attn_sink"], idxs, c["head_dim"] ** -0.5)
        o = torch.cat([o[..., :-rd], rope(o[..., -rd:], fr, inverse=True)], -1)
        b, T = o.shape[:2]
        o = fp8_qdq(o.reshape(b, T, G, -1))
        wo_a = self.w(f"{p}.attn.wo_a.weight").to(BF16).view(G, ol, -1)  # convert.py: bf16
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.linear(o.flatten(2), f"{p}.attn.wo_b.weight")

    def window_kv(self, p, lid, x, fr, sp):
        c, T = self.c, x.size(1)
        win, rd = c["window_size"], c["rope_head_dim"]
        kv = rms(self.linear(x, f"{p}.attn.wkv.weight"), self.W[f"{p}.attn.kv_norm.weight"], self.eps)
        kv = fp8_qdq(torch.cat([kv[..., :-rd], rope(kv[..., -rd:], fr)], -1))
        ring = self.win[lid]
        if sp == 0:
            if T <= win:
                ring[:T] = kv[0]
            else:
                cut = T % win
                ring[cut:win], ring[:cut] = kv[0, -win:].split([win - cut, cut])
            end = torch.arange(T).unsqueeze(1)
            idxs = (end - win + 1).clamp(min=0) + torch.arange(min(T, win))
            idxs = torch.where(idxs > end, -1, idxs)
            return kv, idxs.int().unsqueeze(0)
        if T > 1:
            # a chunk at sp > 0 (a speculative verify window) is the same math as T decode
            # steps: query sp+e sees its window in time order, older rows from the ring as it
            # was before the chunk (rows [0, win)), the chunk's own rows after it
            old = ring.clone()
            q = torch.arange(sp, sp + T).unsqueeze(1)
            j = (q - win + 1).clamp(min=0) + torch.arange(win)
            rows = torch.where(j < sp, j % win, win + j - sp)
            idxs = torch.where(j > q, -1, rows)
            for e in range(T):
                ring[(sp + e) % win] = kv[0, e]
            return torch.cat([old.unsqueeze(0), kv], 1), idxs.int().unsqueeze(0)
        ring[sp % win] = kv[0, 0]
        oldest = sp % win + 1
        idxs = torch.cat([torch.arange(oldest, win), torch.arange(oldest)])
        idxs = torch.where(idxs > sp, -1, idxs)
        return ring.unsqueeze(0), idxs.int().view(1, 1, -1)

    def compressor(self, p, lid, x, sp):
        c, r, T = self.c, self.c["compress_ratios"][lid], x.size(1)
        norm = self.W[f"{p}.attn.compressor.norm.weight"]
        if r == 1:
            return rms(F.linear(x, self.W[f"{p}.attn.compressor.wkv.weight"].to(BF16)), norm, self.eps)
        xf = x.float()
        kv = F.linear(xf, self.W[f"{p}.attn.compressor.wkv.weight"].float())
        sc = F.linear(xf, self.W[f"{p}.attn.compressor.wgate.weight"].float())
        kst, sst = self.cstate[lid]
        if sp == 0:
            do = T >= r
            rem = T % r
            cut = T - rem
            if rem:
                kst[:rem], sst[:rem] = kv[0, cut:], sc[0, cut:]
            kv, sc = kv[:, :cut].unflatten(1, (-1, r)), sc[:, :cut].unflatten(1, (-1, r))
            kv = (kv * sc.softmax(dim=2)).sum(2)
        else:
            do = (sp + 1) % r == 0
            kst[sp % r], sst[sp % r] = kv[0, 0], sc[0, 0]
            if do:
                kv = (kst * sst.softmax(0)).sum(0).view(1, 1, -1)
        if not do:
            return None
        return rms(kv.to(BF16), norm, self.eps)

    def compress_kv(self, p, lid, x, qr, sp, offset):
        c, T = self.c, x.size(1)
        if sp > 0 and T > 1:
            # one position at a time: groups complete where a decode step completes them and
            # each query's top-k sees only the groups complete at its own position
            topk_in, cand_in = self.shared.get("topk"), self.shared.get("cand")
            rows, cands = [], []
            for e in range(T):
                if topk_in is not None:
                    self.shared["topk"] = topk_in[:, e : e + 1]
                if isinstance(cand_in, list):
                    self.shared["cand"] = cand_in[e]
                ckv, idx = self.compress_kv(p, lid, x[:, e : e + 1], qr[:, e : e + 1], sp + e, offset)
                rows.append(idx[0, 0])
                cands.append(self.shared.get("cand"))
            w = max(r.numel() for r in rows)
            idxs = torch.stack([F.pad(r, (0, w - r.numel()), value=-1) for r in rows]).unsqueeze(0)
            self.shared["topk"], self.shared["cand"] = idxs, cands
            return ckv, idxs
        r, rd = c["compress_ratios"][lid], c["rope_head_dim"]
        clen = (sp + T) // r
        latent = None
        is_src = lid < self.n and lid in c["kv_source_layers"]
        if is_src:
            latent = self.compressor(p, lid, x, sp)
            self.shared["ckv"] = self.ckv[lid]
        if lid < self.n and lid in c["index_source_layers"]:
            if clen == 0:
                idxs = torch.empty(1, T, 0, dtype=torch.int32)
            else:
                idxs = self.indexer(p, lid, x, qr, latent, sp, offset)
            self.shared["topk"] = idxs
        else:
            idxs = self.shared["topk"]
        if latent is not None:
            fr = self.freqs[lid][: T - T % r : r] if sp == 0 else self.freqs[lid][sp + 1 - r].unsqueeze(0)
            lat = torch.cat([latent[..., :-rd], rope(latent[..., -rd:], fr)], -1)
            lat = fp4_qdq(lat, 16, e4m3_scale=True)
            self.ckv[lid][sp // r : sp // r + lat.size(1)] = lat[0]
        return self.shared["ckv"][:clen], idxs

    def indexer(self, p, lid, x, qr, latent, sp, offset):
        c, T = self.c, x.size(1)
        r, rd, end = c["compress_ratios"][lid], c["rope_head_dim"], sp + T
        nh, ihd = c["index_n_heads"], c["index_head_dim"]
        fr_all = self.freqs[lid]
        owns = lid in c["kv_source_layers"]
        if owns and latent is not None:
            fr = fr_all[: T - T % r : r] if sp == 0 else fr_all[sp + 1 - r].unsqueeze(0)
            k = rms(F.linear(latent, self.W[f"{p}.attn.indexer.wk.weight"].to(BF16)),
                    self.W[f"{p}.attn.indexer.k_norm.weight"], self.eps)
            k = fp4_qdq(torch.cat([k[..., :-rd], rope(k[..., -rd:], fr)], -1), 32)
            self.idxk[lid][sp // r : sp // r + k.size(1)] = k[0]
        if owns:
            self.shared["idxk"] = self.idxk[lid]
        q = self.linear(qr, f"{p}.attn.indexer.wq_b.weight").unflatten(-1, (nh, ihd))
        q = fp4_qdq(torch.cat([q[..., :-rd], rope(q[..., -rd:], fr_all[sp:end])], -1), 32)
        ik = self.shared["idxk"][: end // r]
        wts = F.linear(x, self.W[f"{p}.attn.indexer.weights_proj.weight"].to(BF16)) * (ihd**-0.5 * nh**-0.5)
        score = torch.einsum("bshd,td->bsht", q, ik)
        score = (score.relu() * wts.unsqueeze(-1)).sum(dim=2)  # [1, T, t]
        if sp == 0:
            lens = (torch.arange(1, T + 1) // r).unsqueeze(-1)
            score = score.masked_fill(torch.arange(T // r) >= lens, -math.inf)
        else:
            lens = end // r
        src = c["candidate_source_layer"]
        if lid == src:
            self.shared["cand"] = self.candidates(score, lens)
        elif 0 <= src < lid:
            score = score.masked_fill(~self.shared["cand"], -math.inf)
        k = min(c["index_topk"], end // r)
        idx = topk_stable(score, k).sort(-1).values
        return torch.where(idx < lens, idx + offset, -1).int()

    def candidates(self, logits, lens):
        bs, kb = self.c["candidate_block_size"], self.c["candidate_topk_blocks"]
        width = logits.size(-1)
        sc = F.pad(logits.float(), (0, -width % bs), value=-math.inf).unflatten(-1, (-1, bs)).amax(-1)
        nb = sc.size(-1)
        last = (lens - 1) // bs
        sc = sc.masked_fill(torch.arange(nb) == last, math.inf)
        k = min(kb, nb)
        ti = topk_stable(sc, k)
        tv = torch.gather(sc, -1, ti)
        keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, ti, tv > -math.inf)
        return keep.repeat_interleave(bs, dim=-1)[..., :width]

    # moe

    def moe(self, p, x, n_experts, topk):
        c = self.c
        D, lim = c["dim"], c["swiglu_limit"]
        xs = x.view(-1, D)
        sc = F.linear(xs.float(), self.W[f"{p}.ffn.gate.weight"].float())
        sc = F.softplus(sc).sqrt()  # sqrtsoftplus; the config's only scoring
        ids = topk_stable(sc + self.W[f"{p}.ffn.gate.bias"], topk)
        wts = sc.gather(1, ids)
        if c["norm_topk_prob"] and topk > 1:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c["route_scale"]
        y = torch.zeros_like(xs, dtype=torch.float32)
        for e in range(n_experts):
            rows, slot = torch.where(ids == e)
            if rows.numel():
                y[rows] += self.expert(f"{p}.ffn.experts.{e}", xs[rows], wts[rows, slot, None], lim)
        y += self.expert(f"{p}.ffn.shared_experts", xs, None, lim)
        return y.to(x.dtype).view(x.shape)

    def expert(self, p, x, wts, lim):
        g = self.linear(x, f"{p}.w1.weight").float()
        u = self.linear(x, f"{p}.w3.weight").float()
        if lim > 0:
            u = u.clamp(-lim, lim)
            g = g.clamp(max=lim)
        h = F.silu(g) * u
        if wts is not None:
            h = wts * h
        return self.linear(h.to(BF16), f"{p}.w2.weight")

    # hyper-connections and blocks

    def hc_mixes(self, p, kind, x):
        xf = x.flatten(2).float()
        rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.W[f"{p}.hc_{kind}_fn"]) * rs
        c = self.c
        return sinkhorn(mixes, self.W[f"{p}.hc_{kind}_scale"], self.W[f"{p}.hc_{kind}_base"], self.hc,
                        c["hc_sinkhorn_iters"], c["hc_eps"])

    @staticmethod
    def hc_pre(x, pre):
        return (pre.unsqueeze(-1) * x.float()).sum(2).to(x.dtype)

    @staticmethod
    def hc_post(x, res, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * res.unsqueeze(-2)).sum(2)
        return y.to(x.dtype)

    def block(self, p, lid, x, sp, pre_mix, attn_fn):
        c = self.c
        n_e = c["n_routed_experts"] if lid < self.n else (c["dspark_n_routed_experts"] or c["n_routed_experts"])
        k_e = c["n_activated_experts"] if lid < self.n else (c["dspark_n_activated_experts"] or c["n_activated_experts"])
        res = x
        a_pre, a_post, a_comb = self.hc_mixes(p, "attn", x)
        h = rms(self.hc_pre(x, pre_mix), self.W[f"{p}.attn_norm.weight"], self.eps)
        h = attn_fn(h)
        x = self.hc_post(h, res, a_post, a_comb)
        res = x
        f_pre, f_post, f_comb = self.hc_mixes(p, "ffn", x)
        h = rms(self.hc_pre(x, a_pre), self.W[f"{p}.ffn_norm.weight"], self.eps)
        h = self.moe(p, h, n_e, k_e)
        x = self.hc_post(h, res, f_post, f_comb)
        return x, f_pre

    def engram(self, p, lid, x, hash_ids):
        c, D, hc = self.c, self.c["dim"], self.hc
        table = self.w(f"{p}.engram.embed.weight")
        rows = table[hash_ids.long()].to(BF16)  # [T, cols, ehd]
        kv = self.linear(rows.flatten(-2).unsqueeze(0), f"{p}.engram.wkv.weight")
        key, val = kv.split([hc * D, D], -1)
        key = key.float().unflatten(-1, (hc, D))
        wt = self.W[f"{p}.engram.q_weight"].float() * self.W[f"{p}.engram.k_weight"].float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * wt * key).sum(-1) * rstd * D**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * val.float().unsqueeze(-2)).to(x.dtype)

    def forward(self, ids, sp, hooks=None):
        """ids [T] -> (last-position logits [V], target hiddens [1, T, D*n]). hooks(name, t)."""
        c = self.c
        hooks = hooks or (lambda n, t: None)
        hashes = self.hash(ids, sp) if self.hash is not None else None
        if hashes is not None:
            hooks("hash", hashes)
        h = self.W["embed.weight"][ids].unsqueeze(0).to(BF16)
        h = h.unsqueeze(2).repeat(1, 1, self.hc, 1)
        pre_mix = torch.zeros(1, ids.numel(), self.hc)
        pre_mix[..., 0] = 1
        mains = []
        for i in range(self.n):
            p = f"layers.{i}"
            if i in c["engram_layer_ids"]:
                h = self.engram(p, i, h, hashes[:, c["engram_layer_ids"].index(i)])
            if i in c["dspark_target_layer_ids"]:
                mains.append(h.mean(dim=2))
            h, pre_mix = self.block(p, i, h, sp, pre_mix, lambda a, p=p, i=i: self.attention(p, i, a, sp))
            hooks(f"layer{i}", h[0])
        h = self.hc_pre(h, pre_mix)
        logits = F.linear(rms(h, self.W["norm.weight"], self.eps)[:, -1].float(), self.W["head.weight"].float())
        return logits[0], (torch.cat(mains, -1) if mains else None)

    # dspark

    def dspark_attention(self, p, lid, x, sp, main_x):
        c = self.c
        H, hd, rd, win = c["n_heads"], c["head_dim"], c["rope_head_dim"], c["window_size"]
        T = main_x.size(1)
        mkv = rms(self.linear(main_x, f"{p}.attn.wkv.weight"), self.W[f"{p}.attn.kv_norm.weight"], self.eps)
        mkv = fp8_qdq(torch.cat([mkv[..., :-rd], rope(mkv[..., -rd:], self.freqs[lid][sp : sp + T])], -1))
        ring = self.win[lid]
        if sp == 0:
            if T <= win:
                ring[:T] = mkv[0]
            else:
                cut = T % win
                ring[cut:win], ring[:cut] = mkv[0, -win:].split([win - cut, cut])
            return x
        B = x.size(1)
        fr = self.freqs[lid][sp + T : sp + T + B]
        qr = rms(self.linear(x, f"{p}.attn.wq_a.weight"), self.W[f"{p}.attn.q_norm.weight"], self.eps)
        q = self.linear(qr, f"{p}.attn.wq_b.weight").unflatten(-1, (H, hd))
        q = torch.cat([q[..., :-rd], rope(q[..., -rd:], fr)], -1)
        kv = rms(self.linear(x, f"{p}.attn.wkv.weight"), self.W[f"{p}.attn.kv_norm.weight"], self.eps)
        kv = fp8_qdq(torch.cat([kv[..., :-rd], rope(kv[..., -rd:], fr)], -1))
        idxs = torch.cat([torch.arange(min(win, sp + T)), win + torch.arange(B)]).int().view(1, 1, -1).expand(1, B, -1)
        for i in range(T):  # positions sp..sp+T (one at decode, the accepted ones after a verify)
            ring[(sp + i) % win] = mkv[0, i]
        kv = torch.cat([ring.unsqueeze(0), kv], 1)
        return self.attn_out(p, q, kv, idxs, fr)

    def forward_spec(self, token, main_hidden, sp):
        """token: the id just sampled; main_hidden [1, T, D*n]. Prefill (sp 0) only seeds each
        stage's window; a decode step returns (draft ids [block+1], logits [block, V], conf)."""
        c = self.c
        n_mtp, blk = c["n_mtp_layers"], c["dspark_block_size"]
        main_x = rms(self.linear(main_hidden, "mtp.0.main_proj.weight"), self.W["mtp.0.main_norm.weight"], self.eps)
        ids = torch.full((blk,), c["dspark_noise_token_id"], dtype=torch.long)
        ids[0] = token
        h = self.W["embed.weight"][ids].unsqueeze(0).to(BF16).unsqueeze(2).repeat(1, 1, self.hc, 1)
        pre_mix = torch.zeros(1, blk, self.hc)
        pre_mix[..., 0] = 1
        for s in range(n_mtp):
            lid, p = self.n + s, f"mtp.{s}"
            if sp == 0:
                self.dspark_attention(p, lid, h, 0, main_x)
                continue
            h, pre_mix = self.block(p, lid, h, sp, pre_mix, lambda a, p=p, lid=lid: self.dspark_attention(p, lid, a, sp, main_x))
        if sp == 0:
            return None
        last = f"mtp.{n_mtp - 1}"
        x = self.hc_pre(h, pre_mix)
        logits = F.linear(rms(x, self.W[f"{last}.norm.weight"], self.eps).float(), self.W["head.weight"].float())[0]
        out = torch.empty(blk + 1, dtype=torch.long)
        out[0] = token
        membs = []
        for i in range(blk):
            memb = self.W[f"{last}.markov_head.embed.weight"][out[i]].to(BF16)
            logits[i] += F.linear(memb.float(), self.W[f"{last}.markov_head.head.weight"].float())
            membs.append(memb)
            out[i + 1] = int(logits[i].argmax())  # temperature 0
        conf = F.linear(torch.cat([x[0], torch.stack(membs)], -1).float(),
                        self.W[f"{last}.confidence_head.proj.weight"].float()).squeeze(-1)
        return out, logits, conf
