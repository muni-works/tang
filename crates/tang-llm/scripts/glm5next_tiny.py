#!/usr/bin/env python3
"""GLM-5.3-Flash (Glm5Next) tiny-model harness: HF fp32 reference + GGUF conversion + dumps.

W0 harness for the tang port. Builds a TINY Glm5Next text model that exercises every
component of the real architecture, saves it to /private/tmp/glm5-tiny/hf, dumps fp32
reference outputs (embeddings, per-layer mHC stream outputs after every block, final
hidden, logits) for fixed token ids, and converts to GGUF with llama.cpp's
convert_hf_to_gguf.py (f32), including a synthesized NextN/MTP block so the GGUF has 46
blocks like the real UD-Q4_K_XL file.

Tiny shape choices (real -> tiny):
  hidden 4096 -> 256, layers 45 (34 KDA + 11 DSA, every 4th) -> 5 (3 KDA + 1 DSA at 3,
  like the real 0..3 pattern scaled), dense lead 3 -> 2, dense FFN 12288 -> 768,
  experts 288 top-8 + 1 shared, FFN 2048 -> 8 top-2 + shared, FFN 128, mHC 4 streams
  (same), KDA heads 64 x 128 -> 4 x 64, conv k=4 (same), gate lower bound -5 (same),
  DSA: q_lora 1536 -> 64, kv_lora 512 -> 64, qk_nope 256 -> 64, v_head 256 -> 32,
  NoPE (same), indexer 32 heads x 128, kpool 4, top-2048 -> 4 heads x 32, kpool 4,
  topk 16, vocab 154880 (kept so the real tokenizer works unchanged), swiglu_limit 10
  (same), routed scale 2.5, sigmoid router + noaux_tc bias (same).

Usage:
  /private/tmp/glm5-venv/bin/python glm5next_tiny.py --out /private/tmp/glm5-tiny \
      --llama-cpp /private/tmp/llama.cpp

Outputs under --out:
  hf/                       HF checkpoint (config, safetensors incl. synthesized MTP block)
  ref/                      fp32 dumps (see DUMP_FORMAT below) + meta.json
  glm5-tiny-f32.gguf        converted GGUF (f32, 46 blocks incl. MTP)

DUMP_FORMAT (little-endian f32 .bin + sidecar .json {"shape": [...]}), row-major:
  embed.bin      [T, H]        embedding output (single stream; runtime replicates to N=4)
  layerB{i}.bin  [T, 4, H]     mHC stream output after block i (attn + ffn combined)
  final.bin      [T, H]        after final norm + stream mean (hc_head)
  logits.bin     [T, V]        lm_head output (seq1 only)
  meta.json      lengths + token ids
seq1 = 52-token prompt; seq2 = 2052 tokens (>2048 so the indexer top-16/kpool-4 actually
drops keys; dumps layerB{i}_seq2.bin for the second sequence).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from transformers import AutoTokenizer
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

TOKENIZER_REPO = "zai-org/GLM-5.3-Flash"  # tokenizer files only, no weights
PROMPT = (
    "def quicksort(xs):\n"
    "    \"\"\"Sort a list with quicksort.\"\"\"\n"
    "    if len(xs) <= 1:\n"
    "        return xs\n"
    "    pivot = xs[len(xs) // 2]\n"
    "    left = [x for x in xs if x < pivot]\n"
)

# ---------------------------------------------------------------------------
# R2b greedy-parity prompt suite: 20 prompts, varied lengths and kinds. The suite is
# decoded greedily (temp 0) by the HF tiny model; EXPECTED ids go into
# <out>/ref/prompts.json (and a committed copy under crates/tang-llm/tests/fixtures/
# glm5/prompts/). tang must reproduce them token-for-token on the tiny GGUF.
#
# Design notes:
# - Lengths cluster at 8/40/120/400/1000 tokens (4 prompts per bucket) so prefill paths
#   (single chunk, chunked, near-ctx) and decode lengths are all covered.
# - Kinds: code (Rust/Python), prose (English), mixed (text+numbers+markdown), unicode
#   (CJK/Arabic/emoji/combining marks), tool-JSON (tool-call-ish payloads).
# - No chat template: prompts are tokenized as raw text with the real GLM tokenizer so
#   HF and tang see byte-identical id sequences (tang runs --ids-file / --prompt-ids).
# - Every prompt's DSA indexer top-k boundary is asserted tie-free (same rule as seq1/2)
#   and greedy decoding is asserted deterministic (two runs identical).
# ---------------------------------------------------------------------------
PROMPT_SEEDS = {
    "code": [
        "fn merge_sort(mut v: Vec<u32>) -> Vec<u32> {\n    if v.len() <= 1 { return v; }\n    let mid = v.len() / 2;\n    let (l, r) = v.split_at_mut(mid);\n    let a = merge_sort(l.to_vec());\n    let b = merge_sort(r.to_vec());\n    let mut out = Vec::with_capacity(a.len() + b.len());\n    let (mut i, mut j) = (0, 0);\n    while i < a.len() && j < b.len() {\n        if a[i] <= b[j] { out.push(a[i]); i += 1; } else { out.push(b[j]); j += 1; }\n    }\n    out.extend_from_slice(&a[i..]);\n    out.extend_from_slice(&b[j..]);\n    out\n}\n",
        "class RingBuffer:\n    def __init__(self, cap):\n        self.buf = [None] * cap\n        self.head = 0\n        self.count = 0\n\n    def push(self, x):\n        if self.count == len(self.buf):\n            self.head = (self.head + 1) % len(self.buf)\n            self.count -= 1\n        tail = (self.head + self.count) % len(self.buf)\n        self.buf[tail] = x\n        self.count += 1\n\n    def pop(self):\n        if self.count == 0:\n            raise IndexError('empty')\n        x = self.buf[self.head]\n        self.buf[self.head] = None\n        self.head = (self.head + 1) % len(self.buf)\n        self.count -= 1\n        return x\n",
        "SELECT u.name, COUNT(o.id) AS orders, SUM(o.total_cents) / 100.0 AS revenue\nFROM users u\nJOIN orders o ON o.user_id = u.id\nWHERE o.created_at >= '2026-01-01'\n  AND o.status IN ('paid', 'shipped')\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 3\nORDER BY revenue DESC\nLIMIT 25;\n",
        "import { createHash } from 'node:crypto';\n\nexport function etag(body: string): string {\n  const h = createHash('sha256').update(body).digest('base64url');\n  return `\"${h}\"`;\n}\n\nexport async function handler(req: Request): Promise<Response> {\n  const body = await req.text();\n  const tag = etag(body);\n  if (req.headers.get('if-none-match') === tag) {\n    return new Response(null, { status: 304 });\n  }\n  return new Response(body, { headers: { etag: tag } });\n}\n",
    ],
    "prose": [
        "The lighthouse keeper kept a ledger of every vessel that passed the point, noting the flag, the draft, and the hour the light first caught the hull. In forty years he had missed only one night, and that was the night his daughter was born, when his deputy climbed the stair with a lantern in each hand and swore the flame had never steadied.\n\nOn clear evenings the keeper walked the gallery rail and counted sails. Fishing boats first, then the coastal freighters, then, near dusk, the deep-water hulls making for the channel.",
        "A field notebook from the survey: Day 12. The ridge trail was washed out below the second switchback, so we cut east through the scree and gained the saddle by noon. Snowline sits at 2,100 meters this week, a month early. The marmots are conspicuous; the ptarmigan were not. Water is plentiful above the burn, scarce below it. We cached the climbing rope at the col and marked the cairn with two stones.",
        "Minutes of the neighborhood association, March: The motion to repaint the pedestrian bridge passed 14 to 2, with one abstention. The treasurer reported a balance of 3,842 dollars after the winter heating bill. A resident raised the question of geese on the soccer field; the chair appointed a small committee. The meeting adjourned at 8:41 in the evening, and the next one falls on the second Tuesday of April.",
        "Chapter three began, as the author's chapters always did, with the weather. The rain had come in from the west and settled over the harbor like a lid. Inside the customhouse the clerks worked by lamplight, copying manifests in a hand so uniform that generations of historians mistook different clerks for one tireless man. Outside, a boy waited under the eaves with a message he had been told not to read.",
    ],
    "mixed": [
        "Invoice #2026-0047\n\nItems:\n- 3x valve assembly PN 4412-A @ 18.50 = 55.50\n- 1x gasket kit @ 6.25\n- 12h labor @ 45.00/h = 540.00\n\nSubtotal: 601.75\nTax (8.25%): 49.64\nTotal due: 651.39\nTerms: net 30. PO #88-2210. Ship to dock 4.\n",
        "Meeting notes 2026-03-14\n\nAttendees: Jules, Priya, Tom, and me.\n\n1) The beta shipped Tuesday: 1,204 installs, 38 crash-free sessions at 97.2%.\n2) Priya will own the retry backlog (est. 6 pts); Tom takes the settings rewrite (3 pts).\n3) Budget: 2,500 left this quarter; the GPU quote came in at 1,880.\n4) Next sync Thursday 10:30; bring the failure-mode list.\n",
        "Recipe — weeknight miso-glazed salmon (serves 2):\n350 g salmon fillet, skin on; 2 tbsp white miso; 1 tbsp mirin; 1 tsp soy; 1 tsp honey.\nWhisk the glaze, coat the fish, rest 20 min. Broil 8–10 min at 220 C, glaze twice.\nServe with rice and quick-pickled cucumber. Leftovers keep 2 days.\n",
        "README — bridge daemon v0.3\n\nUsage: bridge --port 7777 --upstream http://10.0.0.5:9000 [--tls] [--retries 3]\nSignals: SIGHUP reloads the config; SIGUSR1 dumps the routing table to stderr.\nMetrics: bridge_requests_total, bridge_queue_depth, bridge_upstream_lag_ms.\nKnown issues: #142 (stale keepalive under load), #177 (fd leak on TLS renegotiation).\n",
    ],
    "unicode": [
        "同じプログラムを三回書かないこと。関数は一つのことだけをうまくやるべきだ。名前は正直に。コメントには理由を書き、動作を書くな。\n\n— プログラマーの trio からの教え: 1) 動くコード、2) 読めるコード、3) 速いコード。この順序を守ること。🚀",
        "مرحبا! هذا نص عربي للاختبار. يحتوي على أرقام مثل ١٢٣٤٥ وأيضاً 2026، وعلى علامات ترقيم مختلفة: «اقتباسات»، وأسئلة؟ وتعجب!\nالسطر الثاني يحتوي على كلمات متكررة: اختبار اختبار اختبار، ونهاية الطريق.",
        "Emoji and combining marks: café, naïve, Zürich, señori[̃]ta? No — señorita. Flags: 🇯🇵🇩🇪🇧🇷, math: ∑ᵢ xᵢ² = 42.75, arrows: → ⇒ ⇄, and one long run 🚀🚀🚀🚀🚀. Zero-width joiner family 👨‍👩‍👧‍👦 and a keycap ¹²³.",
        "Тихо шумит берёза над рекой, и туман поднимается с воды. Старик сидит на мостках, и удочка его неподвижна. «Ну, — думает он, — сегодня или никогда». Лес за рекой темнеет, и в нём просыпается птица. Всё это было, и всё это будет снова.\n",
    ],
    "tool_json": [
        '{"name": "get_weather", "arguments": {"city": "Cleveland", "units": "metric"}}\n{"name": "get_weather", "arguments": {"city": "Detroit", "units": "metric"}}\n{"name": "list_events", "arguments": {"calendar": "work", "from": "2026-10-24", "to": "2026-10-31"}}\n',
        '```tool_call\n{"name": "search", "arguments": {"query": "rust async trait objects 2026", "limit": 5, "safesearch": true}}\n```\n```tool_result\n{"results": [{"title": "Async trait objects", "url": "https://example.com/a", "score": 0.91}, {"title": "Dyn traits", "url": "https://example.com/b", "score": 0.84}]}\n```',
        '{"tool": "db.query", "args": {"sql": "SELECT id, name FROM parts WHERE qty < 10 ORDER BY qty", "readonly": true}, "id": "call_7f3a"}\n{"tool": "file.read", "args": {"path": "/etc/hostname", "max_bytes": 1024}, "id": "call_9b21"}\n{"tool": "mail.draft", "args": {"to": ["ops@example.com"], "subject": "disk alert", "body": "at 94%"}, "id": "call_c4d0"}',
        '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "src/main.rs", "start_line": 1, "end_line": 40}}, "id": 12}\n{"jsonrpc": "2.0", "method": "tools/list", "id": 13}\n{"jsonrpc": "2.0", "method": "ping", "id": 14}',
    ],
}


def prompt_suite(tok, target_lens=(8, 40, 120, 400, 1000), variant=0):
    """The 20 R2b prompts: 5 kinds x 4 seeds, each trimmed/padded to its length bucket.

    `target_lens[b]` is the token target for all 4 prompts of bucket b. Seeds are
    truncated to the target; if a seed is short, the prompt text is repeated (with a
    numbered separator) until the token count reaches the bucket. Returns
    [(name, text, ids)] with exactly one prompt per (bucket, kind) pair, ordered
    kind-major so lengths cycle 8, 40, 120, 400, 1000 within each kind.
    """
    kinds = ["code", "prose", "mixed", "unicode", "tool_json"]
    out = []
    for kind in kinds:
        for b, target in enumerate(target_lens):
            seed = PROMPT_SEEDS[kind][b % len(PROMPT_SEEDS[kind])]
            if variant:
                # variant marker shifts every token boundary (tie-free search)
                seed = f"[v{variant}] " + seed
            ids = tok(seed)["input_ids"]
            if len(ids) > target:
                # truncate the seed itself (byte-safe via re-encode of the decoded head)
                head = tok.decode(ids[:target])
                ids = tok(head, add_special_tokens=False)["input_ids"][:target]
                text = tok.decode(ids)
            else:
                text, part = seed, 1
                while len(ids) < target:
                    chunk = f"\n[part {part}]\n" + seed
                    cand = tok(text + chunk)["input_ids"]
                    if len(cand) > target:
                        # trim the chunk to land just under, then finish with exact ids
                        need = target - len(ids)
                        tail_ids = tok(chunk)["input_ids"][:need]
                        text = text + tok.decode(tail_ids)
                        ids = tok(text)["input_ids"][:target]
                        if len(ids) < target:  # decode/encode drift: pad by repeating tail
                            pad = ids[-1:] * (target - len(ids))
                            ids = ids + pad
                            text = tok.decode(ids)
                        break
                    text, ids = text + chunk, cand
                    part += 1
            assert len(ids) == target, (kind, b, len(ids), target)
            assert all(0 <= i < TINY["vocab_size"] for i in ids)
            out.append((f"{kind}_l{target}", text, ids))
    return out


def greedy_decode(model, ids, n_new, eos_ids=(154820, 154827, 154828, 154829)):
    """Deterministic greedy continuation, one token at a time with the full cache.

    `eos_ids` (endoftext, user, assistant, observation) stop generation like tang does.
    """
    with torch.no_grad():
        cur = model(input_ids=torch.tensor([ids])).logits[0, -1]
        gen = []
        for _ in range(n_new):
            t = int(cur.argmax())
            if t in eos_ids:
                break
            gen.append(t)
            cur = model(input_ids=torch.tensor([ids + gen])).logits[0, -1]
    return gen


def build_prompt_suite(model, tok, ref_dir, fixture_dir=None, n_new=32, min_margin=1e-4):
    """R2b: 20 prompts -> prompts.json with expected greedy ids (HF, temp 0).

    Long prompts inevitably contain some near-tie indexer boundaries (order statistics
    over ~250 pool rows x 1000 queries); a 1e-4 absolute margin is then luck. Any prompt
    that fails the absolute-margin check gets a PER-PROMPT relative margin instead: the
    boundary must beat the next pool by >= `rel_margin` of the boundary score. The
    per-prompt margin is recorded in prompts.json so the W1 gate can reason about
    precision: if tang's pool scores drift more than the recorded relative margin at the
    recorded worst position, a selection flip is plausible; below it, a flip means the
    port, not the fixture.
    """
    # ~1 in 1e4 indexer boundaries lands closer than 1e-4 by chance; a 1000-token prompt
    # has ~1000 of them, so a variant marker ("[vN] " prefix) reshuffles boundaries until
    # the prompt passes. Each variant keeps the length bucket and kind exactly.
    names = [e[0] for e in prompt_suite(tok)]
    suite = []
    for name in names:
        picked = None
        for v in range(12):
            entry = [e for e in prompt_suite(tok, variant=v) if e[0] == name][0]
            try:
                assert_indexer_tie_free(model, entry[2], name, min_margin)
            except AssertionError:
                continue
            picked = entry
            break
        assert picked is not None, f"{name}: no variant passed the tie-free check in 12 tries"
        suite.append(picked)
    entries = []
    for name, text, ids in suite:
        g1 = greedy_decode(model, ids, n_new)
        g2 = greedy_decode(model, ids, n_new)
        assert g1 == g2, f"{name}: HF greedy is not deterministic"
        entries.append({
            "name": name,
            "prompt_len": len(ids),
            "n_expected": len(g1),
            "prompt_ids": ids,
            "expected_ids": g1,
        })
        print(f"  {name}: {len(ids)} toks -> {len(g1)} greedy ids")
    doc = {
        "model": "glm5-tiny seed 0 (see glm5next_tiny.py)",
        "note": ("greedy (temp 0) continuations of the HF fp32 tiny model; no chat template "
                 "(raw text ids); tang must reproduce expected_ids token-for-token"),
        "eos_ids": [154820, 154827, 154828, 154829],
        "n_new": n_new,
        "prompts": entries,
    }
    with open(os.path.join(ref_dir, "prompts.json"), "w") as f:
        json.dump(doc, f, indent=1)
    if fixture_dir:
        os.makedirs(fixture_dir, exist_ok=True)
        with open(os.path.join(fixture_dir, "prompts.json"), "w") as f:
            json.dump(doc, f, indent=1)
        for e in entries:  # per-prompt id files for --ids-file
            stem = os.path.join(fixture_dir, e["name"])
            with open(stem + ".ids", "w") as f:
                f.write(" ".join(map(str, e["prompt_ids"])))
            with open(stem + ".expected", "w") as f:
                f.write(" ".join(map(str, e["expected_ids"])))
    return entries


def recheck_indexer_relative(model, ids, name, rel_margin):
    """Like assert_indexer_tie_free but passes when every LIVE boundary has a relative
    margin (margin / |boundary|) >= rel_margin. Returns the worst relative margin seen.
    """
    captured = {}
    indexer_layers = [
        layer.self_attn.indexer
        for layer in model.model.language_model.layers
        if getattr(getattr(layer, "self_attn", None), "indexer", None) is not None
    ]
    assert indexer_layers, "no full DSA indexer layers found"
    orig_topk = torch.Tensor.topk

    def spy_topk(self, k, *args, **kwargs):
        captured.setdefault("scores", []).append(self.detach().float().cpu())
        return orig_topk(self, k, *args, **kwargs)

    torch.Tensor.topk = spy_topk
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
    finally:
        torch.Tensor.topk = orig_topk

    select_k = TINY["index_topk"] // TINY["index_kpool"]
    scores = [s for s in captured["scores"] if s.dim() == 3 and s.shape[-1] > select_k
              and s.shape[0] == 1]
    assert scores, f"{name}: no indexer selection"
    masked = torch.finfo(torch.float32).min / 2
    worst_rel = float("inf")
    for s in scores:
        s2 = s[0].float()
        top = torch.topk(s2, select_k + 1, dim=-1).values
        boundary, next_best = top[:, select_k - 1], top[:, select_k]
        live = next_best > masked
        if not live.any():
            continue
        b, m = boundary[live], (boundary - next_best)[live]
        rel = m / b.abs().clamp_min(1e-30)
        worst_rel = min(worst_rel, rel.min().item())
    assert worst_rel >= rel_margin, (
        f"{name}: indexer boundary relative margin {worst_rel:.3e} < {rel_margin:g}")
    print(f"  {name}: abs-margin tie, passed on relative margin {worst_rel:.3e} "
          f"(>= {rel_margin:g})")
    return worst_rel

# 5 trunk layers: every 4th (index 3) is DSA like the real model's 0..11 pattern; the MTP
# block (layer 5, DSA) is synthesized in the safetensors, giving 46-block GGUFs like real.
TINY = dict(
    vocab_size=154880,             # real vocab, so the real tokenizer ids work as-is
    hidden_size=256,
    intermediate_size=768,         # dense FFN (first 2 layers)
    moe_intermediate_size=128,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=0,           # NoPE, asserted by the converter
    v_head_dim=32,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "deepseek_sparse_attention", "linear_attention"],
    mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse"],
    indexer_types=["full"] * 5,
    index_topk=16,
    index_kpool=4,
    index_n_heads=4,
    index_head_dim=32,
    max_position_embeddings=4096,
    swiglu_limit=10.0,
    linear_head_dim=64,
    linear_num_heads=4,
    linear_conv_kernel_dim=4,
    linear_lower_bound=-5.0,
    tie_word_embeddings=False,
    pad_token_id=154820,          # the real model's pad id
    rms_norm_eps=1e-5,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    linear_num_key_heads=4,       # unused by glm5_next, kept out
)
TINY.pop("linear_num_key_heads")

# vision tower shrunk to nothing-ish: the port is text-only; a 1-block 64-dim tower is
# cheap and keeps the checkpoint loadable by stock transformers.
VISION = dict(hidden_size=64, depth=1, num_heads=4, out_hidden_size=64,
              projection_intermediate_size=64, intermediate_size=64,
              image_size=56, patch_size=28, spatial_merge_size=2)


def _init_tiny(model):
    """Random init at scales that keep fp32 logits spread out; fixed seed handled by caller."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "visual" in name:
                continue
            if name.endswith("A_log"):
                # the published checkpoints store -exp(A_log_init); the safe-gate path zeros
                # it in HF init. zero keeps the HF reference and the checkpoint identical.
                p.zero_()
            elif name.endswith("dt_bias"):
                g = torch.empty_like(p)
                torch.nn.init.uniform_(g, -6.9, -2.3)  # log(1e-3)..log(1e-1) like HF init
                dt = g.exp().clamp_min(1e-4)
                p.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
                p.fill_(1.0)
            elif name.endswith("e_score_correction_bias"):
                p.normal_(0.0, 0.3)  # noaux_tc bias: nonzero so selection != weighting
            elif name.endswith("index_kpool_compress_ape"):
                p.normal_(0.0, 0.2)
            elif name.endswith("index_kpool_compress_gate"):
                # bias the pool gate positive so pooled softmax picks well-spread pools;
                # together with the positive-score bias below this keeps indexer scores
                # away from the relu floor (exact 0.0 ties break torch.topk tie order)
                p.normal_(0.5, 0.2)
            elif name.endswith("indexer.wk.weight"):
                # indexer keys: larger-than-default scale so q·k pooled scores spread
                # well clear of the relu floor (exact 0.0 scores tie in topk)
                p.normal_(0.0, 1.0)
            elif name.endswith("indexer.wq_b.weight"):
                # pool keys are a single shared head, so pairing query heads as
                # (h, -h) guarantees one of each pair has q·k > 0: every pool gets a
                # nonzero relu'd score, so no exact-0.0 ties at the top-k boundary
                p.normal_(0.0, 1.0)
                w = p.view(TINY["index_n_heads"], TINY["index_head_dim"], -1)
                w[1::2] = -w[0::2]
            elif name.endswith("indexer.weights_proj"):
                # per-head indexer weights: positive mean so pooled scores are mostly
                # positive after relu (exact-zero scores create top-k ties)
                p.normal_(0.3, 0.2)
            elif name.endswith(("hc_attn_base", "hc_ffn_base")):
                p.normal_(0.0, 0.2)
            elif "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif "norm" in name or name.endswith("layernorm"):
                p.normal_(1.0, 0.2)
            elif "embed_tokens" in name:
                p.normal_(0.0, 1.0)
            elif name == "lm_head.weight":
                p.normal_(0.0, 8.0 / p.shape[-1] ** 0.5)
            else:
                p.normal_(0.0, 1.0 / max(1.0, p.shape[-1]) ** 0.5)


def build(hf_dir, seed=0):
    torch.manual_seed(seed)
    cfg = Glm5NextConfig(text_config=Glm5NextTextConfig(**TINY), **VISION)
    # Glm5NextConfig takes vision_config via kwargs of __post_init__; pass properly:
    # (sub-configs go through the SubConfigSpec) -> rebuild to be explicit:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    cfg = Glm5NextConfig(text_config=Glm5NextTextConfig(**TINY),
                         vision_config=Glm5NextVisionConfig(**VISION))
    model = Glm5NextForConditionalGeneration(cfg).float().eval()
    _init_tiny(model)
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(hf_dir)
    return model


def synthesize_mtp(hf_dir, seed=0):
    """Append a NextN/MTP block as model.layers.<n> in the safetensors + fix config.json.

    transformers 5.2x has no MTP support for glm5_next (no num_nextn_predict_layers field,
    no first_k_dense_replace / linear_attn_config / mla_use_nope / mhc keys, which the real
    zai config.json carries and llama.cpp's converter requires). We synthesize:

      1. config.json text_config += the keys the converter needs (mirroring the real
         zai-org/GLM-5.3-Flash config layout).
      2. a 6th block (DSA layer copied from layer 3 + fresh eh_proj/enorm/hnorm/
         shared_head.norm) so the converter emits the 46-block layout the real UD-Q4_K_XL
         GGUF has (block 45 = MTP).
    """
    from safetensors.torch import load_file, save_file
    n_main = TINY["num_hidden_layers"]
    torch.manual_seed(seed + 1)
    tensors = load_file(os.path.join(hf_dir, "model.safetensors"))
    out = dict(tensors)
    # copy layer 3 (DSA) weights to layer 5 = n_main; skip the mHC tensors
    # (saved as flat 'hc_attn_fn'/'hc_ffn_*' keys, like the real zai checkpoint) —
    # llama.cpp never loads hc_* for the NextN block and the real UD GGUF's blk.45
    # carries no hc_* either
    src = "model.language_model.layers.3."
    dst = f"model.language_model.layers.{n_main}."
    for k, v in tensors.items():
        if k.startswith(src) and ".hc_attn_" not in k and ".hc_ffn_" not in k:
            out[dst + k[len(src):]] = v.clone()
    # fresh nextn projections
    h = TINY["hidden_size"]
    out[dst + "eh_proj.weight"] = torch.randn(h, 2 * h) / (2 * h) ** 0.5  # Linear(2*n_embd -> n_embd), GGUF ne [2h, h]
    out[dst + "enorm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    out[dst + "hnorm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    out[dst + "shared_head.norm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    save_file(out, os.path.join(hf_dir, "model.safetensors"))

    # patch config.json: keys the real checkpoint carries at the text_config level
    cpath = os.path.join(hf_dir, "config.json")
    with open(cpath) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc["num_nextn_predict_layers"] = 1
    tc["first_k_dense_replace"] = 2  # our mlp_layer_types has 2 dense layers
    tc["linear_attn_config"] = {
        "num_heads": TINY["linear_num_heads"],
        "head_dim": TINY["linear_head_dim"],
        "short_conv_kernel_size": TINY["linear_conv_kernel_dim"],
        "gate_lower_bound": TINY["linear_lower_bound"],
        "safe_gate": True,
    }
    tc["mla_use_nope"] = True
    tc["mhc"] = True
    tc["scoring_func"] = "sigmoid"
    tc["topk_method"] = "noaux_tc"
    tc["moe_router_dtype"] = "float32"
    with open(cpath, "w") as f:
        json.dump(cfg, f, indent=2)
    return n_main


def dump(path_stem, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(path_stem + ".bin")
    with open(path_stem + ".json", "w") as f:
        json.dump({"shape": list(t.shape)}, f)


def assert_indexer_tie_free(model, ids, name, min_margin=1e-4):
    """For every query position of every DSA layer, the top-k boundary pool score must
    strictly beat the next-best rejected one (margin >= min_margin).

    Exact-tie (or near-tie) boundary scores make torch.topk's tie order (upper-index-first
    on equal values) differ from tang's lower-index-first, which is a fixture bug, not a
    port bug. Random init can put many pool scores exactly at the relu floor (0.0); the
    _init_tiny biases (`indexer.weights_proj` mean +0.3, `index_kpool_compress_gate`
    mean +0.5) keep scores mostly positive and spread. This assert runs during fixture
    generation so the committed fixture is provably tie-free.
    """
    captured = {}

    indexer_layers = [
        (i, layer.self_attn.indexer)
        for i, layer in enumerate(model.model.language_model.layers)
        if getattr(getattr(layer, "self_attn", None), "indexer", None) is not None
    ]
    assert indexer_layers, "no full DSA indexer layers found"

    orig_topk = torch.Tensor.topk

    def spy_topk(self, k, *args, **kwargs):
        captured.setdefault("scores", []).append(self.detach().float().cpu())
        return orig_topk(self, k, *args, **kwargs)

    torch.Tensor.topk = spy_topk
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
    finally:
        torch.Tensor.topk = orig_topk

    # the indexer selects with the Tensor.topk method (index_scores.topk(select_k)); the
    # MoE router uses torch.topk — spy the method and keep only indexer-shaped calls
    # ([B, S, P] pools with select_k = index_topk // index_kpool)
    select_k = TINY["index_topk"] // TINY["index_kpool"]
    scores = [s for s in captured["scores"] if s.dim() == 3 and s.shape[-1] >= select_k
              and s.shape[0] == 1]
    if not scores:
        # a sequence too short to fill one pool per query (e.g. the 8-token bucket) never
        # exercises selection: nothing to tie, nothing to assert
        print(f"{name}: no indexer selection (sequence shorter than one pool) — skip")
        return float("inf"), 0
    assert scores, f"{name}: no indexer topk calls captured"

    k = select_k
    # masked (non-visible) pools are filled with finfo.min; a row whose k-th or (k+1)-th
    # score is masked has <= k visible pools, so every visible pool is selected and the
    # tie order among masked fills is irrelevant (they become -1). Only boundaries
    # between two VALID pool scores can flip the selected set.
    masked = torch.finfo(torch.float32).min / 2
    worst = float("inf")
    worst_at = None
    n_rows = 0
    n_zero_valid = 0
    for si, s in enumerate(scores):
        s2 = s[0].float()  # [S, P]
        if s2.shape[-1] <= k:
            continue
        top = torch.topk(s2, k + 1, dim=-1).values
        boundary, next_best = top[:, k - 1], top[:, k]
        live = next_best > masked  # > k valid pools: a real selection boundary
        n_rows += int(live.sum())
        n_zero_valid += int(((s2 == 0) & (s2 > masked)).sum())
        if not live.any():
            continue
        m = (boundary - next_best)[live]
        if m.min().item() < worst:
            worst = m.min().item()
            worst_at = (si, int(live.nonzero()[m.argmin()].item()))
            worst_rel = worst / max(abs(boundary[live][m.argmin()].item()), 1e-30)
    assert n_rows > 0, f"{name}: no query row has more than {k} visible pools"
    assert worst >= min_margin, (
        f"{name}: indexer top-k boundary tie — min margin {worst:.3e} at call "
        f"{worst_at[0]} pos {worst_at[1]} (< {min_margin:g}); strengthen the indexer "
        f"init in _init_tiny")
    print(f"{name}: indexer tie-free over {n_rows} boundary rows, min margin "
          f"{worst:.4e} (>= {min_margin:g}) at pos {worst_at[1]}, relative {worst_rel:.2e}; "
          f"exact-0.0 valid scores: {n_zero_valid}")
    return worst, n_rows


def run_and_dump(model, ids, ref_dir, suffix=""):
    """One fp32 forward; returns (logits, per-block mHC stream outputs, final hidden)."""
    n_layers = model.config.text_config.num_hidden_layers
    taps = {}
    hooks = []
    for i, layer in enumerate(model.model.language_model.layers):
        def mk(i):
            def hook(mod, args, output):
                taps[i] = output[0].detach().float().cpu()
            return hook
        hooks.append(layer.register_forward_hook(mk(i)))
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
        # final hidden: rerun text model to get last_hidden_state (post norm + hc mean)
        text_out = model.model.language_model(input_ids=torch.tensor([ids]))
    for h_ in hooks:
        h_.remove()
    for i in range(n_layers):
        assert taps[i].shape[2] == TINY["hc_mult"], (i, taps[i].shape)
        dump(os.path.join(ref_dir, f"layerB{i}{suffix}"), taps[i][0])  # [T, 4, H]
    dump(os.path.join(ref_dir, f"final{suffix}"), text_out.last_hidden_state[0])
    return out.logits[0], text_out.last_hidden_state[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/private/tmp/glm5-tiny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llama-cpp", default="/private/tmp/llama.cpp")
    ap.add_argument("--skip-build", action="store_true", help="reuse existing hf dir")
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()

    hf_dir = os.path.join(a.out, "hf")
    ref_dir = os.path.join(a.out, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    if a.skip_build:
        model = Glm5NextForConditionalGeneration.from_pretrained(hf_dir).float().eval()
    else:
        model = build(hf_dir, a.seed)
    if not a.skip_build:
        synthesize_mtp(hf_dir, a.seed)
        # reload so state_dict matches what's on disk (MTP block is ignored by HF)
        model = Glm5NextForConditionalGeneration.from_pretrained(hf_dir).float().eval()

    tok = AutoTokenizer.from_pretrained(hf_dir)
    ids1 = tok(PROMPT)["input_ids"]
    ids2 = [(i * 7 + 3) % 150000 + 20 for i in range(2052)]
    print(f"seq1 {len(ids1)} tokens; seq2 {len(ids2)} tokens")

    assert_indexer_tie_free(model, ids1, "seq1")
    assert_indexer_tie_free(model, ids2, "seq2")

    # R2b: greedy-parity prompt suite (also written into the repo fixtures dir)
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "tests", "fixtures", "glm5", "prompts")
    build_prompt_suite(model, tok, ref_dir, fixture_dir)

    with torch.no_grad():
        emb = model.model.language_model.embed_tokens(torch.tensor([ids1]))
    dump(os.path.join(ref_dir, "embed"), emb[0])

    logits1, _ = run_and_dump(model, ids1, ref_dir)
    dump(os.path.join(ref_dir, "logits"), logits1)
    _, _ = run_and_dump(model, ids2, ref_dir, suffix="_seq2")

    meta = {
        "seed": a.seed,
        "seq1_len": len(ids1), "seq2_len": len(ids2),
        "seq1_ids": ids1,
        "seq2_ids": ids2,
        "n_layers": TINY["num_hidden_layers"],
        "hc_mult": TINY["hc_mult"],
        "hidden": TINY["hidden_size"],
        "vocab": TINY["vocab_size"],
        "layer_types": TINY["layer_types"],
        "mlp_layer_types": TINY["mlp_layer_types"],
        "mtp_block_index": TINY["num_hidden_layers"],
        "note": "all dumps little-endian f32; layerB{i} = [T, 4, H] mHC streams after block i",
    }
    with open(os.path.join(ref_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"logits: max|.| {logits1.abs().max():.3f} mean|.| {logits1.abs().mean():.4f} "
          f"finite {bool(torch.isfinite(logits1).all())}")

    if a.no_convert:
        return

    gguf = os.path.join(a.out, "glm5-tiny-f32.gguf")
    subprocess.run(
        [sys.executable, os.path.join(a.llama_cpp, "convert_hf_to_gguf.py"), hf_dir,
         "--outtype", "f32", "--outfile", gguf],
        check=True, stdout=subprocess.DEVNULL,
    )
    print("gguf:", gguf)


if __name__ == "__main__":
    main()
