"""GLM-5.3-Flash mHC fixture generator (W2).

Builds a `Glm5NextTextHyperConnection` + `Glm5NextTextHyperHead` on a tiny config with
random weights (fixed seed), runs the reference math in fp32 on fixed inputs, and dumps
raw little-endian f32 .bin files + a shapes JSON into
crates/tang-llm/tests/fixtures/glm5/mhc/.

Reference code: transformers/models/glm5_next/modeling_glm5_next.py (lines 218-336).
"""

import json
import struct
import sys
from pathlib import Path

import torch

from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextHyperConnection,
    Glm5NextTextHyperHead,
)

torch.manual_seed(20261022)

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# A tiny but fully representative text config: hc_mult 4, small hidden. eps/iters at the
# real model's values (1e-5 RMSNorm eps is the model's rms_norm_eps; hc_eps 1e-6;
# 20 Sinkhorn iterations).
cfg = Glm5NextTextConfig(
    hidden_size=64,
    rms_norm_eps=1e-5,
    hc_mult=4,
    hc_eps=1e-6,
    hc_sinkhorn_iters=20,
)

hc = Glm5NextTextHyperConnection(cfg)
head = Glm5NextTextHyperHead()

# Random weights, fp32, matching _init_weights *style* but with a fixed seed and wider
# base/scale so sigmoid/softmax/Sinkhorn are exercised away from their trivial points.
with torch.no_grad():
    hc.fn.normal_(mean=0.0, std=0.02)
    hc.base.normal_(mean=0.0, std=0.5)
    hc.scale.copy_(torch.tensor([0.8, 1.1, 1.3]))
head.to(torch.float32)
hc.to(torch.float32)

# Inputs: the residual streams [B, S, N, D] entering one HC block. Two sequences of
# different lengths, tokens drawn from a normal; also a second independent block's
# worth (like attn_hc then ffn_hc would see different stream states).
def streams(b, s, sd):
    g = torch.Generator().manual_seed(b * 1000 + s)
    return (torch.randn(b, s, cfg.hc_mult, cfg.hidden_size, generator=g) * sd).float()

inputs = [streams(1, 5, 1.0), streams(1, 13, 0.7), streams(2, 7, 1.4)]

# Block outputs y [B, S, D] that the post/comb combine sees.
def block_out(b, s):
    g = torch.Generator().manual_seed(b * 77 + s)
    return (torch.randn(b, s, cfg.hidden_size, generator=g) * 2.0).float()

blockouts = [block_out(1, 5), block_out(1, 13), block_out(2, 7)]

def dump(name, t):
    t = t.detach().to(torch.float32).contiguous()
    assert t.dtype == torch.float32
    (OUT / f"{name}.bin").write_bytes(t.numpy().tobytes(order="C"))
    return list(t.shape)

index = {"model": "glm5-next mHC", "hc_mult": cfg.hc_mult, "hidden_size": cfg.hidden_size,
         "rms_norm_eps": cfg.rms_norm_eps, "hc_eps": cfg.hc_eps,
         "hc_sinkhorn_iters": cfg.hc_sinkhorn_iters,
         "layout": "little-endian raw f32; streams [B,S,N,D] contiguous, D fastest; "
                   "comb [B,S,src,dst] (src-major, dst fastest)",
         "tensors": []}

def reg(name, shape):
    index["tensors"].append({"name": name, "file": f"{name}.bin", "dtype": "f32",
                             "shape": shape})

# Weights (shared between the three cases; one HC block).
for name, t in [("fn", hc.fn), ("base", hc.base), ("scale", hc.scale)]:
    reg(name, dump(name, t))

# Per case: input streams, y, and every intermediate + output.
for i, (x, y) in enumerate(zip(inputs, blockouts)):
    p = f"c{i}_"
    b, s, n, d = x.shape
    reg(f"{p}streams", dump(f"{p}streams", x))
    reg(f"{p}y", dump(f"{p}y", y))

    # --- invariants of the module, recorded so the Rust tests can check them directly
    flat = x.view(b, s, -1).float()
    flat = hc.input_norm(flat)
    mixes = torch.nn.functional.linear(flat, hc.fn.float())
    reg(f"{p}flat_norm", dump(f"{p}flat_norm", flat))
    reg(f"{p}mixes", dump(f"{p}mixes", mixes))

    post, comb, collapsed = hc(x)
    reg(f"{p}post", dump(f"{p}post", post))        # [B,S,N]
    reg(f"{p}comb", dump(f"{p}comb", comb))        # [B,S,N,N] view src-major? see below
    reg(f"{p}collapsed", dump(f"{p}collapsed", collapsed))  # [B,S,D]

    # The full decoder-layer combine: out = post·y + combᵀ @ residual  ([B,S,N,D]).
    out = post.to(torch.float32).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
        comb.to(torch.float32).transpose(-1, -2), x
    )
    reg(f"{p}out", dump(f"{p}out", out))           # [B,S,N,D]

# Final head: unweighted mean over the streams of the last case's output.
mean = head(inputs[0])
reg("head_mean", dump("head_mean", mean))          # [B,S,D]

(OUT / "index.json").write_text(json.dumps(index, indent=1))
print("wrote", OUT)
for t in index["tensors"]:
    print(" ", t["name"], t["shape"])
