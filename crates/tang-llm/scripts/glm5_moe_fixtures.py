"""HF fixtures for the GLM-5.3-Flash MoE CPU reference (crates/tang-llm/src/flash/glm5/moe.rs).

Run with the shared venv (transformers with glm5_next):
    /private/tmp/glm5-venv/bin/python crates/tang-llm/scripts/glm5_moe_fixtures.py [--tiny /private/tmp/glm5-tiny/hf]

Writes raw little-endian f32 .bin files + fixtures.json (name -> file, shape) into
crates/tang-llm/tests/fixtures/glm5/moe/. Two kinds:
- synthetic: tiny random-weight HF modules (router, experts, shared MLP, dense MLP, whole MoE),
  including a swiglu_limit=1 case (so the clamps bite) and a group-limited router (n_group=4).
- tiny_*: the real layer-0 dense MLP and layer-2 MoE of W0's tiny GLM-5 checkpoint, run on a fixed
  input; the Rust test loads the same layers from the tiny GGUF via MoeParams::from_gguf.
"""
import argparse
import json
import os

import torch
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextConfig,
    Glm5NextTextExperts,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
    Glm5NextTextTopkRouter,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures", "glm5", "moe")

HID, NE, K, FF, FF_SH, T = 64, 16, 4, 32, 64, 12

meta = {}


def init(t, g, gain=1.5):
    """Fan-in scaled normal init (O(1) activations, like the real model after ffn_norm), so a
    1e-4 absolute tolerance measures our error rather than HF's own fp32 rounding."""
    t.copy_(torch.randn(t.shape, generator=g) / t.shape[-1] ** 0.5 * gain)


def save(name, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    fn = name.replace(".", "_") + ".bin"
    with open(os.path.join(OUT, fn), "wb") as f:
        f.write(a.tobytes())
    meta[name] = {"file": fn, "shape": list(a.shape)}


def cfg(limit=10.0, n_group=1, topk_group=1, hid=HID, ne=NE, k=K, ff=FF, ff_dense=FF_SH * 4):
    return Glm5NextTextConfig(
        hidden_size=hid, moe_intermediate_size=ff, intermediate_size=ff_dense,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=ne, num_experts_per_tok=k, n_shared_experts=1,
        n_group=n_group, topk_group=topk_group, norm_topk_prob=True,
        routed_scaling_factor=2.5, swiglu_limit=limit, hidden_act="silu",
        mlp_layer_types=["sparse"], layer_types=["full_attention"],
    )


def synthetic(g):
    x = torch.randn(T, HID, generator=g)
    save("x", x)
    c = cfg()

    router = Glm5NextTextTopkRouter(c).eval()
    with torch.no_grad():
        init(router.weight, g, 3.0)
        router.e_score_correction_bias.copy_(torch.randn(NE, generator=g) * 0.3)
        _, w, idx = router(x)
    save("router.weight", router.weight)
    save("router.bias", router.e_score_correction_bias)
    save("router_out.weights", w)
    save("router_out.indices", idx)

    # experts driven by that router's own selection
    experts = Glm5NextTextExperts(c).eval()
    with torch.no_grad():
        init(experts.gate_up_proj, g)
        init(experts.down_proj, g)
        y = experts(x, idx, w)
    save("experts.gate_up_proj", experts.gate_up_proj)
    save("experts.down_proj", experts.down_proj)
    save("experts_out", y)

    # same router, swiglu_limit = 1 so both clamps are active
    experts1 = Glm5NextTextExperts(cfg(limit=1.0)).eval()
    with torch.no_grad():
        init(experts1.gate_up_proj, g)
        init(experts1.down_proj, g)
        y1 = experts1(x, idx, w)
    save("experts_l1.gate_up_proj", experts1.gate_up_proj)
    save("experts_l1.down_proj", experts1.down_proj)
    save("experts_l1_out", y1)

    # group-limited router (n_group 4, topk_group 2): the brief's "if set" path
    rg = Glm5NextTextTopkRouter(cfg(n_group=4, topk_group=2)).eval()
    with torch.no_grad():
        init(rg.weight, g, 3.0)
        rg.e_score_correction_bias.copy_(torch.randn(NE, generator=g) * 0.3)
        _, wg, ig = rg(x)
    save("router_g4.weight", rg.weight)
    save("router_g4.bias", rg.e_score_correction_bias)
    save("router_g4_out.weights", wg)
    save("router_g4_out.indices", ig)

    # shared expert (an ungated Glm5NextTextMLP)
    sh = Glm5NextTextMLP(config=c, intermediate_size=FF_SH).eval()
    with torch.no_grad():
        for p in (sh.gate_proj, sh.up_proj, sh.down_proj):
            init(p.weight, g)
        ys = sh(x)
    save("shared.gate_proj", sh.gate_proj.weight)
    save("shared.up_proj", sh.up_proj.weight)
    save("shared.down_proj", sh.down_proj.weight)
    save("shared_out", ys)

    # whole MoE (router + experts + shared), its own weights
    moe = Glm5NextTextMoE(c).eval()
    with torch.no_grad():
        for _, p in moe.named_parameters():
            init(p, g, 3.0 if p.dim() == 2 and p.shape[0] == NE else 1.5)
        for _, b in moe.named_buffers():
            b.copy_(torch.randn(b.shape, generator=g) * 0.3)
        ym = moe(x)
    for n, p in list(moe.named_parameters()) + list(moe.named_buffers()):
        save("moe." + n, p)
    save("moe_out", ym)

    # dense MLP at limit 10 and limit 1
    for tag, limit in (("dense", 10.0), ("dense_l1", 1.0)):
        d = Glm5NextTextMLP(config=cfg(limit=limit)).eval()
        with torch.no_grad():
            for _, p in d.named_parameters():
                init(p, g)
            yd = d(x)
        for n, p in d.named_parameters():
            save(f"{tag}.{n}", p)
        save(f"{tag}_out", yd)


def tiny(g, hf_dir):
    """Layer 0 (dense) and layer 2 (MoE) of W0's tiny checkpoint, through the HF module classes."""
    from safetensors import safe_open

    cj = json.load(open(os.path.join(hf_dir, "config.json")))
    tc = cj.get("text_config", cj)
    hid, ne, k, ff = tc["hidden_size"], tc["n_routed_experts"], tc["num_experts_per_tok"], tc["moe_intermediate_size"]
    c = cfg(limit=tc["swiglu_limit"], n_group=tc["n_group"], topk_group=tc["topk_group"],
            hid=hid, ne=ne, k=k, ff=ff, ff_dense=tc["intermediate_size"])
    st = safe_open(os.path.join(hf_dir, "model.safetensors"), "pt")
    pre = "model.language_model.layers."
    get = lambda n: st.get_tensor(pre + n).to(torch.float32)

    tt = 8
    x = torch.randn(tt, hid, generator=g)
    save("tiny_x", x)

    dense = Glm5NextTextMLP(config=c).eval()
    with torch.no_grad():
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(dense, n).weight.copy_(get(f"0.mlp.{n}.weight"))
        save("tiny_l0_dense_out", dense(x))

    moe = Glm5NextTextMoE(c).eval()
    with torch.no_grad():
        moe.gate.weight.copy_(get("2.mlp.gate.weight"))
        moe.gate.e_score_correction_bias.copy_(get("2.mlp.gate.e_score_correction_bias"))
        gu = torch.stack([torch.cat([get(f"2.mlp.experts.{e}.gate_proj.weight"),
                                     get(f"2.mlp.experts.{e}.up_proj.weight")]) for e in range(ne)])
        moe.experts.gate_up_proj.copy_(gu)
        moe.experts.down_proj.copy_(torch.stack([get(f"2.mlp.experts.{e}.down_proj.weight") for e in range(ne)]))
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(moe.shared_experts, n).weight.copy_(get(f"2.mlp.shared_experts.{n}.weight"))
        save("tiny_l2_moe_out", moe(x))
        _, w, idx = moe.gate(x)
        save("tiny_l2_router_out.indices", idx)
        save("tiny_l2_router_out.weights", w)


def save_u8(name, a):
    fn = name + ".bin"
    with open(os.path.join(OUT, fn), "wb") as f:
        f.write(a.tobytes())
    meta[name] = {"file": fn, "shape": [int(a.size)], "dtype": "u8"}


def quant(g, gguf_py):
    """Random blocks of each weight type the MoE meets, with finite f16 scales, and llama.cpp's own
    (gguf-py) dequantisation of them: checks tang's and the CUDA decoders on the same bytes."""
    import sys

    import numpy as np

    sys.path.insert(0, gguf_py)
    from gguf import GGMLQuantizationType as Q
    from gguf import quants

    n = 512  # elements: 2 K-quant blocks, 16 Q8_0/Q5_0 blocks
    # (name, ggml id, block elements, block bytes, f16 field offsets inside a block)
    # Q5_0: ggml's high bit for element j + 16 is bit j + 16 of qh — the C mask
    # ((qh >> (j + 12)) & 0x10) reads bit j + 16, not j + 12 (verified against the real tiny
    # GGUF's Q5_0 experts vs HF f32, and bit-exact vs gguf-py). The expected values are built
    # here straight from that formula.
    kinds = [("f16", Q.F16, 1, 2, []), ("q8_0", Q.Q8_0, 32, 34, [0]), ("q5_0", Q.Q5_0, 32, 22, [0]),
             ("q4_k", Q.Q4_K, 256, 144, [0, 2]),
             ("q5_k", Q.Q5_K, 256, 176, [0, 2]), ("q6_k", Q.Q6_K, 256, 210, [208])]
    for name, qt, be, bb, f16s in kinds:
        nb = n // be
        raw = torch.randint(0, 256, (nb * bb,), generator=g, dtype=torch.uint8).numpy().copy()
        if name == "f16":
            raw = (torch.randn(n, generator=g).numpy().astype(np.float16)).view(np.uint8).copy()
        for b in range(nb):
            for off in f16s:
                v = np.float16(0.001 + 0.05 * float(torch.rand(1, generator=g)))
                raw[b * bb + off: b * bb + off + 2] = np.frombuffer(v.tobytes(), dtype=np.uint8)
        if name == "q5_0":
            want = np.empty(n, dtype=np.float32)
            for b in range(nb):
                blk = raw[b * bb: (b + 1) * bb]
                d = float(np.frombuffer(blk[:2].tobytes(), dtype="<f2")[0])
                qh = int(np.frombuffer(blk[2:6].tobytes(), dtype="<u4")[0])
                for j in range(16):
                    want[b * 32 + j] = (((int(blk[6 + j]) & 0xF) | ((qh >> j) & 1) << 4) - 16) * d
                    want[b * 32 + j + 16] = (((int(blk[6 + j]) >> 4) | ((qh >> (j + 16)) & 1) << 4) - 16) * d
        else:
            want = quants.dequantize(raw, qt).astype(np.float32).reshape(-1)
        assert want.size == n and np.isfinite(want).all(), name
        save_u8(f"quant_{name}_raw", raw)
        save(f"quant_{name}_want", torch.from_numpy(want))
        meta[f"quant_{name}_raw"]["ggml_type"] = int(qt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", default="/private/tmp/glm5-tiny/hf")
    ap.add_argument("--gguf-py", default="/private/tmp/llama.cpp/gguf-py")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    g = torch.Generator().manual_seed(5)
    synthetic(g)
    if os.path.isdir(a.tiny):
        tiny(g, a.tiny)
    else:
        print("no tiny checkpoint at", a.tiny, "- skipping tiny_* fixtures")
    if os.path.isdir(a.gguf_py):
        quant(g, a.gguf_py)
    else:
        print("no gguf-py at", a.gguf_py, "- skipping quant_* fixtures")
    with open(os.path.join(OUT, "fixtures.json"), "w") as f:
        json.dump({"dtype": "f32", "endian": "little", "tensors": meta}, f, indent=1)
    print("wrote", len(meta), "tensors to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
