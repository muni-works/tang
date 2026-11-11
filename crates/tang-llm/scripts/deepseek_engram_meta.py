#!/usr/bin/env python3
"""Engram hash constants for a DeepSeek-V4.1 checkpoint: writes `engram.json` next to it.

tang hashes n-grams in Rust but takes these from here rather than recompute them: the
compressed token map comes from normalizing every token's text with the `tokenizers` library,
and the multipliers from numpy's PCG64. A transcription of DeepSeek's `inference/engram.py`
(`build_compressed_token_map`, `compute_hash_multipliers`, `EngramLayout`); the map's size must
equal the config's `engram_compressed_vocab_size`, which checks the normalizer end to end.

Usage:
  /private/tmp/ds41-venv/bin/python deepseek_engram_meta.py --model-dir <dir with config.json,
      tokenizer.json> [--out <dir>/engram.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepseek_ref as R  # noqa: E402


def compressed_token_map(tokenizer_json):
    """Each token id -> an id shared by every token that normalizes alike (NFKC, NFD, strip
    accents, lowercase, collapse whitespace, strip; a lone space survives)."""
    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(tokenizer_json)
    sentinel = ""
    norm = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new, lookup = {}, []
    for tid in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:  # a partial UTF-8 byte token: key it by its raw form
            key = tok.id_to_token(tid)
        else:
            n = norm.normalize_str(text)
            key = n if n else text
        lookup.append(key_to_new.setdefault(key, len(key_to_new)))
    return lookup, len(key_to_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    # DeepSeek's repo keeps the runtime config at inference/config.json
    inf = os.path.join(a.model_dir, "inference", "config.json")
    c = json.load(open(inf if os.path.exists(inf) else os.path.join(a.model_dir, "config.json")))
    c.setdefault("max_seq_len", 1)
    token_map, n = compressed_token_map(os.path.join(a.model_dir, "tokenizer.json"))
    want = c["engram_compressed_vocab_size"]
    assert n == want, f"compressed vocab {n}, config says {want}: the normalizer disagrees"
    # the model's vocab may be wider than the tokenizer's; pad as engram.py's lookup would
    token_map += [0] * (c["vocab_size"] - len(token_map))
    primes = R.engram_primes(c)
    flat = [[p for g in layer for p in g] for layer in primes]
    offsets = [np.cumsum([0, *f[:-1]]).tolist() for f in flat]
    bound = max(1, (np.iinfo(np.int64).max // n) // 2)
    mult = [
        (np.random.default_rng(10007 * lid).integers(0, bound, size=(c["engram_max_ngram_size"],), dtype=np.int64) * 2 + 1).tolist()
        for lid in c["engram_layer_ids"]
    ]
    rows = [sum(f) for f in flat]
    assert rows == c["engram_num_embeddings"], f"bucket rows {rows} != engram_num_embeddings"
    out = a.out or os.path.join(a.model_dir, "engram.json")
    with open(out, "w") as f:
        json.dump({"token_map": token_map, "pad": token_map[c.get("engram_pad_id", 2)],
                   "multipliers": mult, "primes": primes, "offsets": offsets}, f)
    print(f"compressed vocab {n} (matches), table rows {rows} (match) -> {out}")


if __name__ == "__main__":
    main()
