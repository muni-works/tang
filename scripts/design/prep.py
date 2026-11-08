"""Tokenize FineWeb-Edu (sample-10BT, first shards) with the GPT-2 tokenizer
into uint16 train.bin / val.bin. usage: python prep.py OUTDIR [nshards]"""
import os, sys, urllib.request
import numpy as np, pyarrow.parquet as pq, tiktoken
from concurrent.futures import ThreadPoolExecutor

out = sys.argv[1]
nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 2
os.makedirs(out, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
base = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/"
chunks = []
for s in range(nsh):
    f = f"{out}/{s:03d}.parquet"
    if not os.path.exists(f):
        urllib.request.urlretrieve(base + f"{s:03d}_00000.parquet", f)
    texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
    def tok(batch):
        return np.concatenate([np.array([eot] + t, dtype=np.uint16)
                               for t in enc.encode_ordinary_batch(batch, num_threads=8)])
    with ThreadPoolExecutor(16) as ex:
        chunks += list(ex.map(tok, [texts[i:i + 2000] for i in range(0, len(texts), 2000)]))
    print("shard", s, sum(len(c) for c in chunks), flush=True)
    os.remove(f)
allt = np.concatenate(chunks)
nval = 10_000_000
allt[-nval:].tofile(f"{out}/val.bin")
allt[:-nval].tofile(f"{out}/train.bin")
print("tokens train", len(allt) - nval, "val", nval, flush=True)
