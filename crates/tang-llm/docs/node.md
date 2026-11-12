# Node API

Status: implemented (milestone N0 of frog node). What frog's scheduler reads and drives on each
machine that runs `tang-llm serve`. All of it sits behind the same gate as `/v1/*`: beyond
loopback a paired client's key (or the static `TANG_API_KEY`) is required, and an unpaired node
answers `401 {"error": "not paired"}`. Pairing, the node identity and the auth rules are frog's
`docs/node-protocol.md` (sections 1–6); here: `src/auth.rs`, `src/pair.rs`, `src/model_id.rs`.

- `GET /node`: the machine, its model, measured rates, the queue and the KV it holds.
- `POST /models/load`, `POST /models/unload`: swap the model, only into free memory.
- `POST /models/pull`, `DELETE /models/pull`, `POST /models/delete`: download a model from
  frog's weight mirror into the Hugging Face cache, cancel that, remove one (see
  [Downloads](#downloads)).
- `x-frog-priority: interactive | background` on `/v1/chat/completions` and `/v1/prefill`.

## `GET /node`

```json
{
  "schema": 2,
  "node_id": "7950c88da7b7e5bb86e3cc225258b704",
  "identity": { "fingerprint": "3f2a91c0…", "public_key": "<base64 Ed25519 public key>" },
  "name": "cams-mbp",
  "version": "0.1.0",
  "hardware": {
    "kind": "metal",
    "name": "Apple M4 Max",
    "unified_memory": true,
    "total_bytes": 51539607552,
    "free_bytes": 32712736768
  },
  "models": {
    "loaded": [
      {
        "id": "mlx-community/Qwen3-4B-4bit",
        "path": "/Users/cam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb…",
        "context_window": 32768,
        "vision": false,
        "loaded_at": 1791150341,
        "load_secs": 1.10,
        "memory": { "weights_bytes": 2263022529, "kv_bytes": 603979776 },
        "rates": {
          "prefill_tok_s": 794.9, "prefill_samples": 2,
          "decode_tok_s": 352.4, "decode_samples": 1
        },
        "step_cost": {
          "model": "secs = a + f*decode_rows + sum over prefill chunks of (g + b*q + d*q*c + e*q^2), …",
          "a": 8.6, "f": 1.1,
          "small": { "g": 18.8, "b": 539, "d": 317, "e": 0 },
          "large": { "g": 96.6, "b": 795, "d": 357, "e": 0 },
          "samples": 1890
        },
        "kv": {
          "block_positions": 256,
          "memory_blocks": ["09cc3dc1a601afcb", "41c5f0e12cfd940c", "…"],
          "disk_blocks": ["024893f416fca2b3", "31025fec1d68432f", "…"],
          "prompt_cache_keys": [
            { "key": "warm", "cached_tokens": 9480 },
            { "key": "rivers", "cached_tokens": 74 }
          ]
        }
      }
    ],
    "loading": null,
    "on_disk": [
      {
        "id": "mlx-community/Qwen3-4B-4bit",
        "path": "/Users/cam/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb…",
        "size_bytes": 2263022529,
        "load_bytes": 4544723905,
        "loaded": true
      }
    ]
  },
  "queue": {
    "running": [
      {
        "id": 2, "priority": "background",
        "prompt_tokens": 9480, "cached_tokens": 0,
        "prefilled_tokens": 512, "generated_tokens": 0,
        "yields": 0, "queued_ms": 1051, "running_ms": 1047
      }
    ],
    "waiting": [
      { "id": 3, "priority": "interactive", "prompt_tokens": 13, "yields": 0, "queued_ms": 52 }
    ]
  }
}
```

Fields:

- `schema`: the shape's version. New fields don't bump it; a field changing meaning or going
  away does. Clients should ignore fields they don't know.
- Schema 2 adds `identity` (the Ed25519 key in `~/.cache/tang/node-identity` and its
  fingerprint, which clients pin), `name` (the host name), and `weights_hash` and `quant` on
  every loaded and on-disk model (`src/model_id.rs`). `kv.prompt_cache_keys` is shown only to a
  caller the gate let through.
- `node_id`: 128 random bits made the first time and kept in `~/.cache/tang/node-id`
  (`TANG_CACHE_DIR` overrides the directory). It names the machine, not the process: it
  survives restarts and model swaps.
- `hardware.kind`: `metal`, `cuda` or `cpu`. `total_bytes` is VRAM, or RAM where the GPU shares
  it (`unified_memory`).
- `hardware.free_bytes`: what a load may use without taking memory from anything else. On CUDA,
  the driver's free VRAM (every process's use counted). On unified memory, RAM the system has
  available (free and inactive pages; purgeable ones are left out, as only their owner decides
  when they go), and on Metal no more than what's left of the GPU's recommended working set.
  It is not tang's KV budget: that's tang's own cap inside whatever it already holds.
- `models.loaded`: the process's model (one at most, see below). `memory.weights_bytes` is
  estimated from the checkpoint (exact for MLX 4-bit and bf16 checkpoints kept as they are);
  `memory.kv_bytes` is what the KV block pool has allocated (it grows to its budget and isn't
  handed back while the model is loaded).
- `rates`: exponential moving averages (newest sample weighted 0.25) over real requests, never a
  benchmark, `null` until measured. A prefill counts once it computed at least 64 positions (the
  cached ones don't count), a decode once it generated at least 16 tokens; decode includes
  speculative decoding. A background prefill that gave way counts each of its runs.
- `models.loading`: `{id, elapsed_ms}` while a load runs, else `null`.
- `models.pulling`: `{model, done_bytes, total_bytes}` while a pull runs, else `null`.
  `models.last_pull`: `{model, ok, cancelled, error}` for the last pull that ended since the
  server started, else `null`.
- `models.on_disk`: checkpoints in the Hugging Face cache (`HF_HUB_CACHE`, `$HF_HOME/hub` or
  `~/.cache/huggingface/hub`) that have a config, tokenizer and safetensors. `id` is what
  `/models/load` takes. `load_bytes` is what a load must find free: the weights as this server
  keeps them (its `--f32` / `--q4` setting) plus headroom of 8192 positions of KV and 1 GiB of
  scratch.
- `queue.running`, `queue.waiting`: in the order they'll run. `prompt_tokens` is counted when
  the request is queued; `yields` is how often a background request gave way.
- `kv.memory_blocks`, `kv.disk_blocks`: ids of sealed 256-position blocks (16 hex digits, the
  FNV-1a chain of `docs/paged-kv.md`, so the id of block `i` stands for every token up to the end
  of block `i`). A prompt whose block-`i` id is in either list skips prefilling the first
  `(i + 1) * 256` positions here. Ids are per model (they hash tokens only).
- `kv.prompt_cache_keys`: conversations with a cache in memory and the positions each holds.
  With `--kv-ttl` set, blocks, files and keys past the TTL are left out of all three lists
  (see [retention.md](retention.md)).

### A list, not a Bloom filter

The block lists are plain hex. A block is 256 positions of every layer's K and V, which for
the models tang serves is tens of MiB (36 MiB for Qwen3-4B in bf16; ~28 MiB for Qwen3-0.6B; more
for larger models). So the KV budget (half of RAM on a Mac; one or a few contexts on a 24 GB GPU)
holds at most around a thousand blocks, and the disk tier's default 8 GB about 220 for
Qwen3-4B. At ~19 bytes per id in JSON that's 20–90 KB per `/node` even for a large disk
budget, a few ms over Tailscale and nothing to parse. A Bloom filter at 1% false positives
would save most of those bytes but answers only "maybe here", can't say which blocks are gone
when a filter is rebuilt, and needs both sides to agree on its hashing; the exact list lets the
scheduler count exactly how much of a prompt a node holds. Worth revisiting only past ~50k blocks
(a multi-terabyte disk tier).

## Loads

The server holds one model at a time (one engine on one worker thread). The minimal honest
version of loading follows from that:

- `POST /models/load {"model": "<repo id or directory>"}` replaces the current model. It runs
  only if `load_bytes` fits in `hardware.free_bytes` plus what unloading the current model gives
  back (its estimated weights and KV pool). Otherwise **507** with the numbers, and the current
  model stays. A load never evicts another process's memory; it only takes free memory.
- **409** while any request is running or waiting: the scheduler drains a node before swapping.
  A request that slips in after the check fails with "model X was unloaded" rather than running
  on another model (requests are tagged with the model they were accepted for).
- **404** if the model isn't in the cache; **200** `{"loaded", "already": true}` if it's already
  loaded; otherwise **200** `{"loaded", "replaced", "load_secs", "weights_bytes"}`. If the new
  model fails to load, the old one is loaded again and the error is **500**.
- `POST /models/unload` frees the model (weights and KV pool; the KV disk tier is flushed and
  kept). **200** `{"unloaded": <id or null>, "freed_bytes"}`; **409** while busy. With no model,
  `/v1/chat/completions` and `/v1/prefill` answer **503** and `/v1/models` lists nothing.

Loading uses the command line's settings for every model (context, dtype, KV slots and budget,
speculative decoding); KV on disk and the draft store are per model as before.

## Downloads

A node downloads only what it's told to, file by file, from URLs it's given (frog's curated
catalog, served from the `frog-weights` R2 bucket at `weights.frog.toys`; Hugging Face is only
the mirror script's source). `src/pull.rs`.

- `POST /models/pull`:

  ```json
  {
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "revision": "73e3e38d981303bc594367cd910ea6eb48349da8",
    "files": [
      { "path": "config.json", "url": "https://weights.frog.toys/mlx-community/Qwen3-0.6B-4bit/73e3…/config.json",
        "sha256": "…", "size": 937 },
      …
    ]
  }
  ```

  A catalog entry from `scripts/mirror-weights.sh` carries exactly these (`id` is `model`).
  `revision` is optional (without it the snapshot is named from the files' hashes).
  - **202** `{"pulling", "total_bytes"}` once it has started; follow `models.pulling` in
    `/node` and read `models.last_pull` for how it ended.
  - **507** up front if it won't fit: in memory, when the files include `config.json` (fetched
    first, then `node::need` on the listed safetensors bytes against `hardware.total_bytes`,
    the device on its own since a node runs one model), or on disk (bytes still to download
    against the free space of the hub cache's filesystem).
  - **409** while another pull runs (one at a time per node). **400** for a malformed body: the
    model must be `org/name`, file paths relative without `..`, sha256 64 hex digits, URLs http(s).
  - Files land in the Hugging Face cache layout: `blobs/<sha256>` (downloaded to
    `blobs/<sha256>.part` first, checked against its size and sha256, then renamed),
    `snapshots/<revision>/<path>` symlinked to the blob, and `refs/main` written last. A model
    shows up in `on_disk` (and `/models/load` finds it) only once all of that is done.
  - A `.part` file left by a cancelled or failed pull is resumed with `Range: bytes=<n>-` the
    next time the same model is pulled (a server that answers 200 instead of 206 restarts that
    file). A file whose sha256 doesn't match is deleted and fails the pull.
- `DELETE /models/pull` stops the running pull: **200** `{"cancelled": <model>}`, **404** if none.
  What it downloaded stays for a later resume; `last_pull.cancelled` is `true`.
- `POST /models/delete {"model": "<repo id>"}` removes the model's whole directory from the hub
  cache: **200** `{"deleted"}`, **404** if it isn't there, **409** for the loaded or loading
  model (unload first) or the one being pulled (cancel first), **400** for anything but `org/name`.

### Supported families

What the generic path (`/models/load`, not Flash mode) serves. Checked on an RTX 3090 (mew) by
loading each through `/models/load` and running one chat completion.

tang d6a4c7ce, `--features cuda`, `serve --device cuda --ctx 4096`, ~9 GB of the 3090's VRAM free
(other servers held the rest), 2026-11-12. Decode tok/s is the server's `predicted_per_second`
for one 64-token reply (speculative drafting on, the default).

| Family | Checkpoint | Loads | Coherent | Decode tok/s | Notes |
|---|---|---|---|---|---|
| Qwen3 dense, MLX 4-bit | mlx-community/Qwen3-0.6B-4bit | yes, 3.4 s | yes | 447 | 0.34 GB weights. Also pulled from a mirror with `/models/pull` and served on CPU |
| Qwen3 dense, bf16 | Qwen/Qwen3-0.6B | yes, 1.3 s | yes | 328 | 1.5 GB; with `--q4`: 0.42 GB, 424 tok/s |
| Qwen3-VL | Qwen/Qwen3-VL-2B-Instruct | yes, 5.2 s | yes (text) | 183 | 4.3 GB; images not tried here |
| Gemma 3 | unsloth/gemma-3-1b-it | yes, 2.7 s | yes | 211 | 2.0 GB. 4B: bf16 needs 10.8 GB (refused with 9 GB free); `--q4` loads in 43 s, 145 tok/s |
| gpt-oss | openai/gpt-oss-20b | fit check refuses | untested | – | needs 15.2 GB (13.8 GB MXFP4 weights + headroom); not run on a 24 GB-free GPU yet |

Notes: Qwen3 thinks by default, so a short `max_tokens` can go entirely to `reasoning_content`.
`google/gemma-3-*` is gated on Hugging Face; the mirror copies it with its licence (`gemma`).

## Priority

`x-frog-priority: interactive | background` (absent: interactive; anything else: 400) on
`/v1/chat/completions` and `/v1/prefill`.

Requests were already serialized: one worker thread owns the engine and runs requests to
completion from a channel. The channel is now a queue with two lanes (`src/queue.rs`); the
worker always takes the interactive lane first, FIFO within a lane. While a background request
prefills, the engine asks between 512-token prefill chunks whether to go on
(`engine::Control`); it stops if an interactive request is waiting. The full blocks it
prefilled are sealed (so they're shared and survive even if its slot is reused), and the request
goes back to the front of the background lane. Run again, it reuses what it prefilled like any
prompt with a cached prefix, from its own slot or from the pool. The client sees one response;
its `cached_tokens` and prefill rate are reported as of its first run.

So an interactive request waits at most one prefill chunk of a background request, not its
whole prefill. Qwen3-4B on an M4 Max: an interactive turn sent 1 s after a 28.5k-token
background prefill answered in 0.42 s; behind the same prefill sent as interactive (FIFO, as
before) it took 102.5 s.

Not preempted: a background request that's already decoding runs to the end (frog's background
work is mostly prefill-only warming), and a prompt with images is prefilled in one go.

## Open questions for the scheduler (N4)

- **Prompt hashes need the tokenizer.** Block ids hash token ids, so the scheduler must render
  the chat template and tokenize to match `memory_blocks` against a prompt. Options: frog
  tokenizes (one tokenizer per model family), or a node endpoint answers "how much of this
  prompt do you hold" for a candidate prompt.
- **Rates vary with length.** Prefill tok/s falls as the context grows (attention is
  quadratic), and short prefills are overhead-bound. One EMA mixes them. Answered in part by
  `step_cost` (a model of a batched step's time in decode rows and prefill chunks, `q` new
  tokens after `c` cached; see `docs/batching.md`): a prefill of `n` tokens after `c` costs
  about `g + b·n + d·n·c + e·n²` ms (thousands of tokens) in its regime. It's fitted per model
  and device from a calibration at load and the steps the node runs.
- **Background decode isn't preempted.** If frog sends subagents or compaction (which decode)
  as background, an interactive request can wait a whole background generation. Preempting
  decode means keeping sampler and parser state across the gap.
- **Starvation.** Interactive work always wins; a node busy with interactive turns never runs
  background work. Fine on a personal setup; the scheduler should route background work to the
  idle node.
- **Keyless requests take slots by shared prefix.** A request without `prompt_cache_key`
  may take over a background conversation's slot because they share a template prefix; the
  background request then gets its blocks back from the pool, minus its last partial block.
  frog should key everything.
- **One model per process.** Running two models on one machine means two processes on two
  ports; the scheduler would treat them as two nodes sharing `hardware.free_bytes`.
- **Fit check is an estimate.** Weights are estimated from the checkpoint, and the headroom is a
  fixed 8192 positions; a long context can still grow the KV pool past what was free at load
  time (it grows only within tang's KV budget, which is set at start).
