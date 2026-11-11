#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash tiny-model fixture: random weights in the real checkpoint format, plus
golden dumps from `deepseek_ref.py`.

M0 harness for the tang port (docs/deepseek-v41-plan.md). The tiny model exercises every piece
of the real one: sliding window, compressed KV at ratios 2 and 1 with cross-layer sharing,
indexer sharing, two-level candidate top-k, Engram, sqrtsoftplus MoE with fp4 experts, mHC,
and the three-stage DSpark drafter.

Usage:
  /private/tmp/ds41-venv/bin/python deepseek_tiny.py --out /private/tmp/ds41-tiny

Outputs under --out:
  hf/model.safetensors   weights, real naming and dtypes (I8-packed fp4 experts, F8_E4M3 with
                         F8_E8M0 scales, BF16, F32)
  hf/config.json         the inference/config.json keys for this model
  ref/                   little-endian f32 .bin + sidecar .json {"shape": [...]}:
    step{s}_hash           [T, n_engram_layers, n_hash_cols] engram hash ids
    step{s}_in{i}          [T, hc, D] stream into layer i (before its Engram)
    step{s}_pm{i}          [T, hc] the pre-mix layer i's attention collapses with
    step{s}_layer{i}       [T, hc, D] stream after block i
    step{s}_logits         [V] last-position logits
    step{s}_main_hidden    [T, D * n_targets] DSpark target hiddens
    step{s}_spec_ids       [block + 1] DSpark draft ids (decode steps only)
    step{s}_spec_logits    [block, V] DSpark draft logits
    step{s}_spec_conf      [block] DSpark confidence
  ref/meta.json          ids, start position and greedy next id per step
Step 0 is the prefill; steps 1.. feed back the greedy token one at a time; the last step feeds
the last step's DSpark block [token, drafts] as one chunk (a speculative verify window).
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepseek_ref as R  # noqa: E402

TINY = dict(
    max_seq_len=128,
    vocab_size=512,
    dim=256,
    moe_inter_dim=128,
    n_layers=8,
    n_mtp_layers=3,
    n_heads=8,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    norm_topk_prob=True,
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=128,
    head_dim=128,
    rope_head_dim=64,
    norm_eps=1e-20,
    o_groups=4,
    o_lora_rank=64,
    window_size=8,
    # the real pattern scaled down: two window-only layers, a ratio-2 run reading one source, a
    # ratio-1 run reading another, then three window-only DSpark stages
    compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0],
    kv_source_layers=[2, 5],
    index_source_layers=[2, 4, 5, 7],
    original_seq_len=32,
    rope_theta=10000.0,
    rope_factor=4,
    beta_fast=32,
    beta_slow=1,
    compress_rope_theta=160000.0,
    index_n_heads=4,
    index_head_dim=64,
    index_topk=6,
    candidate_source_layer=5,
    candidate_topk_blocks=3,
    candidate_block_size=4,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=[1, 4],
    engram_max_ngram_size=4,
    engram_vocab_size=1000,
    engram_n_heads=2,
    engram_head_dim=64,
    engram_pad_id=2,
    engram_compressed_vocab_size=300,
    dspark_block_size=5,
    dspark_noise_token_id=500,
    dspark_target_layer_ids=[5, 6, 7],
    dspark_markov_rank=32,
    dspark_n_routed_experts=4,
    dspark_n_activated_experts=2,
)


def tiny_token_map():
    """Stands in for the tokenizer-derived compressed map: a fixed many-to-one fold."""
    V, C = TINY["vocab_size"], TINY["engram_compressed_vocab_size"]
    return [(i * 7) % C for i in range(V)], C


def quant_fp8(w, per_row):
    """F8_E4M3 + F8_E8M0 power-of-two scales, per 32x32 block or per row x 32."""
    if per_row:
        z = w.unflatten(1, (-1, 32))
        s = R.pow2_ceil(z.abs().amax(-1).clamp(min=1e-12) / R.FP8_MAX)
        q = (z / s.unsqueeze(-1)).clamp(-R.FP8_MAX, R.FP8_MAX).to(torch.float8_e4m3fn).flatten(1)
        return q, s.to(torch.float8_e8m0fnu)
    z = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))
    s = R.pow2_ceil(z.abs().amax((1, 3)).clamp(min=1e-12) / R.FP8_MAX)
    q = (z / s[:, None, :, None]).clamp(-R.FP8_MAX, R.FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(2, 3).flatten(0, 1), s.to(torch.float8_e8m0fnu)


def quant_fp4(w):
    """e2m1 packed two to an int8 byte along K (low nibble first), E8M0 scale per row x 32."""
    z = w.unflatten(1, (-1, 32))
    s = R.pow2_ceil(z.abs().amax(-1).clamp(min=1e-12) / R.FP4_MAX)
    codes = R.fp4_codes(R.fp4_round(z / s.unsqueeze(-1))).flatten(1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed, s.to(torch.float8_e8m0fnu)


def make_weights(c, gen):
    W = {}
    for name, shape, kind in R.tensor_specs(c):
        sname = name[: -len("weight")] + "scale"
        if kind in ("fp8", "fp8row", "fp4"):
            std = 1.0 if kind == "fp8row" else 1 / math.sqrt(shape[1])
            w = torch.randn(shape, generator=gen) * std
            W[name], W[sname] = quant_fp4(w) if kind == "fp4" else quant_fp8(w, kind == "fp8row")
        elif name.endswith("attn_sink"):
            W[name] = torch.randn(shape, generator=gen)
        elif name.endswith(("hc_attn_fn", "hc_ffn_fn")):
            W[name] = torch.randn(shape, generator=gen) * 0.05
        elif name.endswith(("hc_attn_base", "hc_ffn_base")):
            W[name] = torch.randn(shape, generator=gen) * 0.5
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            W[name] = 0.5 + torch.rand(shape, generator=gen)
        elif name.endswith("gate.bias"):
            W[name] = torch.randn(shape, generator=gen) * 0.1
        elif len(shape) == 1 or name.endswith(("q_weight", "k_weight")):
            W[name] = (1 + 0.1 * torch.randn(shape, generator=gen)).to(torch.bfloat16)
        else:
            W[name] = (torch.randn(shape, generator=gen) / math.sqrt(shape[1])).to(torch.bfloat16)
        if kind == "f32":
            W[name] = W[name].float()
    return W


class Dumper:
    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, name, t):
        a = t.detach().float().cpu().numpy().astype("<f4")
        a.tofile(os.path.join(self.dir, f"{name}.bin"))
        with open(os.path.join(self.dir, f"{name}.json"), "w") as f:
            json.dump({"shape": list(a.shape)}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/private/tmp/ds41-tiny")
    ap.add_argument("--prefill", type=int, default=40)
    ap.add_argument("--decode", type=int, default=8)
    args = ap.parse_args()

    c = dict(TINY)
    c["engram_num_embeddings"] = [sum(sum(g) for g in layer) for layer in R.engram_primes(c)]
    W = make_weights(c, torch.Generator().manual_seed(1234))

    hf = os.path.join(args.out, "hf")
    os.makedirs(hf, exist_ok=True)
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in W.items()}, os.path.join(hf, "model.safetensors"))
    with open(os.path.join(hf, "config.json"), "w") as f:
        json.dump(c, f, indent=1)

    token_map, C = tiny_token_map()
    with open(os.path.join(hf, "engram_token_map.json"), "w") as f:
        json.dump(token_map, f)
    model = R.Model(c, W, token_map, C)
    # the hash constants tang reads instead of recomputing (numpy's PCG64, the tokenizer fold)
    eh = model.hash
    with open(os.path.join(hf, "engram.json"), "w") as f:
        json.dump({"token_map": token_map, "pad": eh.pad, "multipliers": eh.mult.tolist(),
                   "primes": eh.primes.tolist(), "offsets": eh.offsets.tolist()}, f)
    dump = Dumper(os.path.join(args.out, "ref"))

    # each backbone layer's input stream (before Engram) and pre-mix, for teacher forcing
    step_of = {"s": 0}
    block, engram, pending = model.block, model.engram, {}

    def engram_tap(p, lid, x, hs):
        pending[lid] = x[0].clone()
        return engram(p, lid, x, hs)

    def block_tap(p, lid, x, sp, pre_mix, fn):
        if lid < c["n_layers"]:
            s = step_of["s"]
            dump(f"step{s}_in{lid}", pending.pop(lid, x[0]))
            dump(f"step{s}_pm{lid}", pre_mix[0])
        return block(p, lid, x, sp, pre_mix, fn)

    model.block, model.engram = block_tap, engram_tap

    g = torch.Generator().manual_seed(7)
    ids = torch.randint(3, c["vocab_size"], (args.prefill,), generator=g)
    meta, pos = {"steps": []}, 0
    with torch.inference_mode():
        for step in range(args.decode + 2):
            if step == args.decode + 1:
                # the last step is a verify window at sp > 0: the greedy token and its drafts
                ids = spec[0].clone()
            step_of["s"] = step
            logits, main_hidden = model.forward(ids, pos, lambda n, t: dump(f"step{step}_{n}", t))
            dump(f"step{step}_logits", logits)
            dump(f"step{step}_main_hidden", main_hidden[0])
            nxt = int(logits.argmax())
            spec = model.forward_spec(nxt, main_hidden, pos)
            if spec is not None:
                sid, slog, sconf = spec
                dump(f"step{step}_spec_ids", sid)
                dump(f"step{step}_spec_logits", slog)
                dump(f"step{step}_spec_conf", sconf)
            meta["steps"].append({"start_pos": pos, "ids": ids.tolist(), "next_id": nxt,
                                  "spec_ids": spec[0].tolist() if spec else None})
            print(f"step {step}: T={ids.numel()} next={nxt} spec={spec[0].tolist() if spec else None}")
            pos += ids.numel()
            ids = torch.tensor([nxt])
    with open(os.path.join(args.out, "ref", "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
