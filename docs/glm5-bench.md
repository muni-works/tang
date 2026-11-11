# GLM-5.3-Flash all-GPU decode bench

One command, one summary line, every FFN on the GPU. Use it for every decode number on
GLM-5.3-Flash: runs with `TANG_GLM5_GPU_FFN=0` (the H200, which can't hold all 45 FFN
layers in 141 GB) measure the CPU FFN, not the kernels.

```sh
GLM5_FLAGS='TANG_GLM5_MOE=fast TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_MHC_FUSED=1 TANG_GLM5_MLA_RESIDENT=1 TANG_GLM5_GPU_HEAD=1' \
  scripts/glm5-bench.sh root@195.26.233.140 -p 10741          # the B300, direct TCP
```

Output (last line):

```
bench 66aa4b5e TANG_GLM5_MOE=fast ...: ids OK, prefill X tok/s (N tok), decode Y tok/s, embed .. ms kda .. ms dsa .. ms mhc .. ms ffn .. ms head .. ms
```

## What it does

1. rsyncs the source dir (default: the worktree the script lives in; pass another as the
   last argument) to `/root/tang-bench/src`, md5s every `*.rs` + `Cargo.toml` on both sides
   and refuses on a mismatch (the stale-binary guard).
2. `cargo build --release -p tang-llm --bin tang-llm --features cuda` (installs rustup if
   missing). `GLM5_NO_BUILD=1` skips it for back-to-back A/B runs on one tree.
3. checks all six UD-Q4_K_XL shards and fetches them (`hf download
   unsloth/GLM-5.3-Flash-GGUF`, ~186 GB, ~30 min) if any is missing.
4. on the box, detached, under `flock ~/.gpu.lock`, with `TANG_GLM5_GPU_FFN*` unset (all
   45 FFNs resident):
   - **ids gate**: `'Write a haiku about the sea.' -n 8 --temp 0` must print the CPU oracle
     `785 1196 6801 264 6386 38038 911 279`; otherwise the line says `ids MISMATCH` and no
     speed is reported.
   - **timed decode**: `scripts/glm5-bench-prompt.txt` (~200 tokens), `-n 128 --temp 0`,
     `TANG_GLM5_TRACE=1`. prefill / decode tok/s from the `prompt:` / `decode:` lines; the
     stage ms are the **wall ms per decode token** of the top-level rows of the trace's
     decode table (mhc = the four `mhc_pre/post.*` rows summed).
5. polls every 30 s and prints the summary. Logs: `/root/tang-bench/out/<label>/gate.log`,
   `bench.log` (the full trace table is at the end of bench.log).

## Knobs

| env | default | |
|---|---|---|
| `GLM5_FLAGS` | empty (naive kernels) | env assignments for both runs, space-separated, no spaces in values |
| `GLM5_LABEL` | derived from the flags | output dir name under `/root/tang-bench/out` |
| `GLM5_ROOT` | `/root/tang-bench` | remote work dir (src, target, out); use your own on a shared box |
| `GLM5_N` | 128 | decode tokens for the timed run |
| `GLM5_GGUF` | `/workspace/glm53/UD-Q4_K_XL/...-00001-of-00006.gguf` | first shard |
| `GLM5_NO_BUILD` | 0 | 1 = reuse the last build (source md5 still checked) |

## Adding a flag / doing an A/B

A new kernel behind `TANG_GLM5_FOO=1` needs no script change: run twice, same tree, one
flag apart, and compare the two lines.

```sh
B='TANG_GLM5_MOE=fast TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_MHC_FUSED=1 TANG_GLM5_MLA_RESIDENT=1 TANG_GLM5_GPU_HEAD=1'
GLM5_FLAGS="$B"                 scripts/glm5-bench.sh root@HOST -p PORT
GLM5_FLAGS="$B TANG_GLM5_FOO=1" GLM5_NO_BUILD=1 scripts/glm5-bench.sh root@HOST -p PORT
```

Keep every other flag fixed across the pair (e.g. GEMV8 when testing KDA graphs). A
result only counts if its line says `ids OK`.

## Access

Needs direct-TCP ssh (`root@IP -p PORT`, from the RunPod pod's `ssh.direct`). The
`ssh.runpod.io` proxy is PTY-only and rejects commands, rsync and scp. In zsh pass host
and port as separate words (`$H` with spaces does not word-split).

The baseline set is the one above, MLA_RESIDENT and GPU_HEAD included: without
MLA_RESIDENT, DSA alone costs 1.6 s/token. B300 baseline (2026-10-27, 66aa4b5e): 22.9
tok/s decode. Results table: crates/tang-llm/NOTES.md.
