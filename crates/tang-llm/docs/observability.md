# Observability: OpenTelemetry traces and metrics

`tang-llm serve` records per-request traces and per-step metrics behind the cargo feature
`otel`, off by default: the default build has no OpenTelemetry dependencies and every hook in
`src/telemetry.rs` compiles to nothing.

With the feature on, export goes through `crates/tang-otel` (shared with the OpenCL runtime
and `qwen_session`). **The default exporter is a local file** of OTLP/JSON lines: one
`ExportTraceServiceRequest` / `ExportMetricsServiceRequest` per line, the framing the
collector's `otlpjsonfile` receiver reads (the same contract as ipse's trainer). **Nothing
goes over the network** unless asked for with `TANG_OTEL_LIVE=1` or `serve --otel-live`.

```bash
cargo build --release --features cuda,otel   # or metal,otel
tang-llm serve ...                           # -> ~/.cache/tang/otel/tang-llm.otel.jsonl
OTEL_FILE=/srv/otel/tang-llm.otel.jsonl tang-llm serve ...
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4328 tang-llm serve ... --otel-live   # file + live
```

| Variable | Meaning |
|---|---|
| `OTEL_FILE` (or `TANG_OTEL_FILE`) | The JSON-lines file (appended). Unset: `~/.cache/tang/otel/<service>.otel.jsonl`. `off`: no file. |
| `TANG_OTEL_LIVE=1` / `--otel-live` | Also export over OTLP/HTTP (protobuf) to `OTEL_EXPORTER_OTLP_ENDPOINT`. Without it the endpoint is ignored (with a note on stderr). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base URL; `/v1/traces` and `/v1/metrics` are appended. Live export only. |
| `OTEL_EXPORTER_OTLP_HEADERS` | `k=v,k2=v2`, e.g. the backend's API key. |
| `OTEL_SERVICE_NAME` | Defaults to `tang-llm`. |
| `OTEL_RESOURCE_ATTRIBUTES` | Extra resource attributes (`host.name` and `process.pid` are set already). |

`OTEL_FILE=off` with no live export turns it all off: each hook is one check of an unset
`OnceLock`.

Earlier builds exported whenever `OTEL_EXPORTER_OTLP_ENDPOINT` was set; that now also needs
`TANG_OTEL_LIVE=1` (or `--otel-live`).

The other standard `OTEL_EXPORTER_OTLP_*` variables (per-signal endpoints, timeouts) are read by
the exporter too. Spans go through a batch processor and metrics through a 10 s periodic
reader, each on its own thread; neither touches the model thread.

## Traces

`POST /v1/chat/completions` (and `/v1/prefill`) continue an incoming W3C `traceparent`; without
one, a new trace starts. Each request is one server span with four children:

```
chat.completions  (server; request.id, gen_ai.request.model, outcome)
├─ queued    priority, queue.wait_ms, yields
├─ prefill   prompt_tokens, cached_tokens, new_tokens, chunks
├─ decode    completion_tokens, draft_tokens, accepted_draft_tokens, verify_windows,
│            tokens_per_verify_window, shared_streams_avg, tokens_per_second
└─ finish    outcome = ok | cancelled | error
```

- `request.id` is the client's `x-request-id` (the same id as the usage log line);
  `gen_ai.request.model` is the loaded model's name. Both are on every span.
- `queued` runs from arrival to the first time the worker picks the request up. A background
  request that gives way to interactive work and goes back in the queue keeps that span;
  `yields` counts how often it gave way.
- `prefill` ends where decode starts: decode is the last `completion_tokens / decode_tok_s`
  seconds of the run, as the engine measures it.
- `verify_windows` is the decode forwards the stream took; with speculative decoding
  `tokens_per_verify_window` is above 1. `shared_streams_avg` is how many other streams shared
  its forwards, on average (0 alone).
- An error sets the root span's status to error with the message.

Spans are built when the request concludes, from timestamps it carried; there's no span per
decode step.

## Metrics

| Instrument | Kind | Attributes | What |
|---|---|---|---|
| `tang_llm.step.duration` (s) | histogram | `kind` = decode / mixed / prefill | One forward. `mixed` is decode rows plus prefill chunks in one step. |
| `tang_llm.step.rows` | histogram | `kind` | Streams in that forward. |
| `tang_llm.prefill.chunk_tokens` | histogram | | Mean prefill chunk of a step that had any. |
| `tang_llm.prefix.hit_ratio` | histogram | | Per request: cached / prompt tokens. |
| `tang_llm.queue.depth` | gauge | | Requests waiting, sampled between batched steps and when one is picked up. |
| `tang_llm.stream.tokens_per_second` | histogram | | Per request: decode tok/s. |

Step metrics come from `Engine::step`, so they cover batched and single-stream serving alike.

## Overhead

Measured with `examples/bench_batch` (it calls `telemetry::init()`), Qwen3-0.6B-Base bf16 on an
M4 Max (Metal), 256 tokens per stream, five interleaved runs of each build, the `otel` build
exporting to a local OTLP sink. A 0.6B model has the shortest steps, so this is the worst case
for relative overhead.

The hook itself, `bench_batch x --hook-cost` (2M calls of `telemetry::step`):

| build | ns per step |
|---|---|
| default | 0.4 (an empty call) |
| `otel`, exporting | 163–186 |

Against the shortest step measured (5 ms, one 0.6B stream) that is 0.003%. Per request,
`finish` builds five spans once (a few µs, off the per-step path).

End to end, ms/step (median of 5 interleaved runs; min–max):

| batch | default | `otel` |
|---|---|---|
| 1 | 5.54 (4.95–7.68) | 6.05 (5.14–7.45) |
| 4 | 11.71 (10.31–16.15) | 12.02 (10.74–18.18) |
| 8 | 16.31 (15.64–29.36) | 16.82 (15.64–30.26) |

The M4's run-to-run spread (up to 2x within this session) is far wider than 1%, so the
end-to-end runs bound nothing finer than that; the hook timing is the evidence for "under
1%". Rerun on a quiet CUDA box to confirm end to end:

```bash
cargo build --release --example bench_batch --no-default-features --features cuda,otel
TANG_OTEL_LIVE=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 target/release/examples/bench_batch <model> --batches 1,4,8
```

## DeepSeek-V4.1-Flash: the step timeline (`flash/deepseek/obs.rs`)

The DeepSeek engine has its own per-forward timeline that needs no OpenTelemetry: every
forward (`step`) is timed with CUDA events on the model's stream and with host wall-clock,
and on the direct path its interior is spans nested by name. Off by default.

| Variable | Meaning |
|---|---|
| `DS_OBS=1` | One line per forward on stderr (rank 0): `step decode t=1 sp=812 path=graph 38.2ms gpu=35.1ms host=3.1ms launches=0 syncs=1 d2h=.. h2d=.. allocs=0`, and per speculative step `spec drafted=5 accepted=3 draft=..ms verify=..ms rewind=..ms`. |
| `DS_OBS=2` | Also the interior spans, on every `DS_OBS_EVERY`-th forward, and a per-op breakdown (name, ms, % of the step's device time, count) printed for it. |
| `DS_OBS_EVERY=N` | Sampling period for interior spans and OTel forward spans (default 50). |
| `DS_OBS_DIRECT=1` | Decode / verify steps skip CUDA graphs, so their interior can be timed (a profiled run, slower). |
| `DS_OBS_FILE=path` | Every observed forward (every rank) as a JSON line (`StepRec`: kind, t, sp, path, captured, rank, step, parent, wall_ms, gpu_ms, host_ms, reduce_wait_ms, arena_bytes, counts, spans), plus `{"spec": ...}` lines. |

The span tree of a direct step:

```
step{kind: prefill|decode|verify|draft, t, sp, path: graph|direct, captured}
├─ embed
├─ layer{i}                        (one per backbone layer; `layer` field = i)
│  ├─ engram                       (Engram layers)
│  ├─ hc                           (mHC mixes / pre / post and the norms, several per layer)
│  ├─ attn
│  │  ├─ q  window_kv
│  │  ├─ compress ─ indexer        (compressed layers; the indexer on index sources)
│  │  ├─ sparse_attn               (window/top-k index rows and the attention kernel)
│  │  └─ wo                        (un-RoPE, quant, wo_a, wo_b)
│  └─ moe
│     ├─ router  route
│     ├─ grouped_gu  grouped_down  (or `experts`, the per-expert loop)
│     ├─ reduce                    (the EP all-reduce: enqueue to completion)
│     └─ shared  finish
└─ head
```

- **Device time** comes from events recorded at span boundaries and read only once they have
  completed: after the step's own read-back sync, or, on an EP rank that skips the read-back,
  at its next step's start (which syncs anyway). The timeline adds no host sync.
- **Graph steps are opaque.** A replayed graph is one launch; its step is timed around the
  launch and has no interior. No event is recorded inside a capture (the capturing step is
  marked `captured`), so capture works with the timeline on. To see a decode step's
  interior, run with `DS_OBS_DIRECT=1`.
- `host_ms` = `wall_ms - gpu_ms`: host time the device did not overlap (Engram hashing and row
  gather, staging, launch gaps, the read-back). On the direct path a large `host_ms` with a
  high launch count is launch-bound.
- **Counters** are per thread (each EP rank has its own) and always counted (a thread-local
  increment): kernel launches (every `ds_*`/`dg_*` launch goes through `obs::launch`), host
  syncs, D2H and H2D bytes (graph replays count their staged inputs), device / pinned
  allocations (arena and staging growth), graph captures and replays, expert tensors uploaded.
  A step reports its deltas.
- **EP**: each rank records its own steps with `rank`; `parent` is the group's command
  sequence number, equal on every rank, so the records of one forward line up.
  `reduce_wait_ms` is the summed `reduce` spans (direct path only).

Overhead (RTX 3090, tiny fixture, decode µs/step, best of 10 rounds): see the table in
[deepseek-v41-plan.md](deepseek-v41-plan.md#how-to-profile-a-step). Off it is a few relaxed
atomic loads and thread-local increments per step.

### `tang-llm deepseek-profile`

```bash
tang-llm deepseek-profile <checkpoint dir> [--devices 0,1,2,3] [--steps N] [--prompt-tokens N] \
    [--direct] [--graph-debug] [--no-spec] [--ctx N]
```

Loads the model (one GPU, or expert parallel with `--devices`), prefills a synthetic prompt
(`--prompt-tokens`, default 64) and runs `--steps` (default 20) decode / verify forwards,
greedy, with DSpark unless `--no-spec` / `DS_SPEC=0`, at `DS_OBS=2` on every step, then prints:
one row per forward (wall / gpu / host ms, launches, syncs, D2H / H2D bytes, allocs, expert
uploads), per kind the means over its steps after the first (the first uploads weights and
sizes the arena), and for steps with an interior a per-op table (ms per step, % of the step's
device time, count per step, ms each), the share covered by `embed` + layers + `head`, and
other ranks' gpu / reduce-wait means. `--direct` forces the direct path so decode and verify
steps have interiors; `--graph-debug` prints each change of a step shape's path (direct ->
graph+capture -> graph). It also honours `DS_OBS_FILE`, and with `otel` emits its spans under
a `deepseek.profile` root.

### OpenTelemetry (feature `otel`)

`deepseek-serve` requests get the same `chat.completions` > `queued` / `prefill` / `decode` /
`finish` spans as the dense server (the attributes come from the run's `Usage`). Under them:

```
prefill
└─ forward{kind=prefill|draft(seed), ...}
decode
├─ forward{kind, t, sp, path, captured, rank, step, gpu_ms, host_ms, launches, syncs,
│          d2h_bytes, h2d_bytes, allocs, expert_uploads, reduce_wait_ms}
│  ├─ embed / layer / ... / head   (the timeline's spans, when the direct path ran;
│  │                                attributes op = dotted path, layer, gpu_ms, host_ms)
│  └─ forward{rank=1..}            (other EP ranks' forward for the same command)
└─ spec_step{drafted, accepted, draft_ms, verify_ms, rewind_ms}
```

`forward` spans are sampled: every `DS_OBS_EVERY`-th forward (default 50), every forward at
`DS_OBS=2`. With telemetry installed the sampled forwards are timed even at `DS_OBS=0`. Spans are
buffered (at most 20k) and built when the request finishes, like the request's own spans.
EP ranks run on their own threads; their forwards are linked to rank 0's by the command
sequence number each `Cmd` carries (spans are built after the fact, so there is no live span
context to pass).

| Instrument | Kind | Attributes | What |
|---|---|---|---|
| `deepseek.forward.ms` | histogram | `kind`, `path` | An observed forward's device time (rank 0). |
| `deepseek.spec.accepted` | histogram | | Drafts accepted per speculative step. |
| `deepseek.launches` | counter | `kind`, `path` | Kernel launches in observed forwards. |
| `deepseek.syncs` | counter | `kind`, `path` | Host syncs in observed forwards. |
| `deepseek.arena.bytes` | gauge | | The step arena's capacity. |

## GLM-5 engine traces (`TANG_GLM5_TRACE=1`)

The GLM-5.3 engine's twin of `TANG_CODER_PROF` plus the spans above
(`src/flash/glm5/trace.rs`). Every window builds a span tree: `glm5.window` → `embed`,
`layer` (one per layer) → `mhc_pre.attn` | `kda` (`proj`, `gates`, `core`, `out_gate`,
`o_proj`, `sync+read`) or `dsa` (`q_a.host`, `kv_append`, `idx_append`, `idx_select`,
`q_b.host`, `attend.gather`, `attend`, `o_proj.host`) | `mhc_post.attn` | `mhc_pre.ffn` |
`ffn` (`moe.shared`, `moe.router`, `moe.experts`, `moe.combine`, or `dense`; `moe.cpu` /
`dense.cpu` off the GPU) | `mhc_post.ffn`, then `head` → `head.mean`, `head.norm`, `lm_head`.

Each span carries host wall time and, from `tang_moe::gpu::trace`, the GPU time of the
kernels launched inside it (two timing events per launch outside graph capture, resolved at
window end), launch count, H2D/D2H bytes and calls, the host time inside synchronous copies
(the transfer plus the wait it implies) and `cuMemAlloc`/`cuMemFree` calls and time. The
weight GEMVs, the experts and MLA's attend declare the bytes they stream
(`gpu::trace::bytes`), which gives GB/s and %-of-SOL per kernel; SOL is the datasheet
bandwidth for known cards (H200 4800 GB/s) or `TANG_SOL_GBS`.

- **Table:** `flash-generate` prints the end-of-run bottleneck table on stderr: decode per
  token, prefill as totals; every stage, the kernels by GPU time, and the top-level stages
  ranked by wall with host = wall − GPU. Kernels run async, so a stage that only launches
  shows little wall and its GPU time; the wait lands where the stream drains (`sync+read`,
  `moe.combine`'s `sync`).
- **otel:** with `--features otel` the trees go out through `telemetry::glm5_tree`
  (`tang-otel`, OTLP/JSON lines to `OTEL_FILE`), one trace per window, and every stage's GPU
  ms feeds the `tang_llm.glm5.stage.gpu` histogram (`stage`, `phase`).
- **Off** (the default): one relaxed atomic load per hook.

Overhead (`engine_gpu_trace_is_transparent`, the tiny 3-layer f32 fixture where a decode
window is almost all launches — 10.3 ms of host for ~0.1 ms of GPU): traced logits are
bit-identical to untraced, and the window goes 10.342 → 10.371 ms traced, **+0.3%** — the
cost is one event record + a queue push per launch, no per-launch sync (the previous end
event is the next launch's start; `drain` syncs one event per stream).

Real model (GLM-5.3-Flash UD-Q4_K_XL, 1×H200, `TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1
TANG_GLM5_MHC_FUSED=1 TANG_GLM5_GPU_FFN=0 TANG_GLM5_KDA_MAXT=64`, `-n 16`, Oct 26): ids
traced equal untraced and the CPU oracle; decode 382 s traced vs 403 / 409 s for the
untraced runs before and after it (the box ran other jobs, so the traced run is inside the
run-to-run noise). Excerpt of the decode table:

```text
-- decode, per token: 15 windows, 15 tokens, wall 25477.8 ms (0.04 tok/s), GPU 155.7 ms (1% of wall), 724 launches, ...
stage                       calls    wall ms  %wall    gpu ms  launch    H2D MB    D2H MB  copy ms alloc ms     GB/s   %SOL
ffn                            45   23816.52  93.5%      0.00       0     0.000     0.000     0.00     0.00        -      -
  moe.cpu                      42   23028.63  90.4%      0.00       0     0.000     0.000     0.00     0.00        -      -
dsa                            11    1552.48   6.1%     96.85      33     1.081     0.743     2.13     2.13        9    0.2
  attend.gather                11     898.23   3.5%      0.00       0     0.000     0.000     0.00     0.00        -      -
kda                            34      57.20   0.2%     53.66     510     0.557     0.557     1.14     0.00       97    2.0
kernels by GPU time (decode, per token):
mla_attn_naive           dsa/attend                      11     74.27   6752.2  47.7%       10    0.2
k5_beta                  kda/gates                       34     43.54   1280.6  28.0%        0    0.0
k5_gemv8                 kda/proj                       102      2.85     27.9   1.8%     1277   26.6
```

## The OpenCL runtime and `qwen_session` (the K1)

`tang-bench --features opencl,otel` (pulls `tang-gpu/otel`) writes the same file format
(service `qwen_chat` / `qwen_decode`, or whatever the embedding process passes to
`tang_otel::init`; hold its guard so spans flush at exit, or call `tang_otel::flush()`).

```
qwen.prefill        tang.qwen.prompt_tokens, reused_tokens, new_tokens, chunks, chunk_sizes, ms, tokens_per_second
├─ qwen.prefill.chunk   chunk_tokens, kctx, pos, host_ms   (host time; the device syncs at the end)
└─ opencl.run           (sampled, see below)
qwen.decode.step    tang.qwen.pos, tokens, ms               (every TANG_OTEL_DECODE_SAMPLE-th step, default 1)
└─ opencl.run
opencl.build        tang.cl.device, kernels, programs, cache_hits, cache_misses, build_ms_total, wall_ms
└─ opencl.program.compile   (cache misses only) source_bytes, build_ms
qwen.tune           tang.tune.key, budget_s, candidates, chosen
└─ qwen.tune.candidate      kind = decode | prefill, config, ms, gpu_temp_c
```

`opencl.run` is a program run with every kernel's profiling event kept and waited on:
`tang.cl.kernels`, `enqueue_ms` (host), `queued_to_complete_ms` (first queued to last end,
device clock), `device_busy_ms` (sum of kernel start-to-end), `top_kernel` / `top_kernel_ms`.
One run in `TANG_OTEL_CL_SAMPLE` (default 64; 0 = never) is profiled, because waiting on
hundreds of events costs a little; `TANG_OTEL_CL_KERNEL_SPANS=1` adds an `opencl.kernel`
child per kernel (hundreds per step: debugging only).

Metrics: `tang.opencl.program.build_ms` (histogram, `tang.cl.cache` = hit / miss),
`tang.opencl.program.builds` (counter, same attribute), `tang.opencl.run.device_ms` and
`tang.opencl.kernel.device_ms` (sampled histograms).

**Parent context.** `Session::set_trace_parent(Some("00-<trace>-<span>-01"))` (or
`set_trace_ids(trace, span)`) makes every later prefill / decode span, and the OpenCL spans
under them, children of that span: broca sets it per turn to its turn span. `None` goes back
to separate traces. Without the feature it returns `false` and does nothing, so callers
compile either way. `qwen_chat --traceparent TP` exercises it. `Session::open`'s builds run
before a parent can be set; they nest under whatever OpenTelemetry context is current on the
calling thread.

### Overhead (OpenCL decode)

`qwen_chat` (Qwen3-0.6B MLX 4-bit, 3 turns, `--gen 128`, `TANG_BACKEND=opencl`) on an M4 Max
through Apple's OpenCL, 16 interleaved pairs, a default build against `--features
opencl,otel` writing the file with default sampling: decode tok/s median 131.1 (default) vs
132.0 (`otel`), best 140.1 vs 141.3. No measurable cost; the machine was heavily loaded
(load average 150-200) so pairwise spread was wide and nothing finer than ~1% is resolved.
Rerun on the K1 to confirm on the Adreno.

## Tests

`cargo test -p tang-llm --features otel --lib otel` runs a request through the worker with
in-memory span and metric exporters, checking the traceparent continuation, the span tree,
each phase's attributes and that every metric is exported. `cargo test -p tang-otel` checks
the file exporter's line format, the remote-parent continuation, and that an endpoint alone
does not turn on network export. The default `cargo test` runs without either.

---

## Appendix: zest (edge → router → tang-llm) follow-up spec

zest is a separate private repo; this is what it needs so one trace spans the hops. Nothing
here is built yet. The user picks the backend (Honeycomb or Grafana Cloud) and creates its
account and keys.

### Shape of a trace

```
edge  zest.request            (Worker, root unless the client sent traceparent)
├─ edge  auth                 KV key lookup
├─ edge  admit                Account DO admit
└─ edge  router.call          fetch to the router; traceparent → router
   └─ router  chat            (axum, continues edge's traceparent)
      ├─ router  admit        tier limits
      ├─ router  upstream.connect   until tang-llm's response headers
      ├─ router  upstream.ttfb      until the first body chunk
      ├─ router  stream             first chunk → end of relay
      └─ tang-llm chat.completions  (traceparent forwarded; spans above)
```

Every span carries `zest.request_id` (the edge's `requestId`, already sent to the router and
on to tang-llm as `x-request-id`, so tang-llm's `request.id` matches), plus `zest.tier`, and
on the edge `zest.key_id` (never the key). Don't put prompts or completions in spans.

### Edge (Cloudflare Worker, `edge/`)

Workers' native tracing (open beta, Oct 2026) exports OTLP to an account-level destination
and auto-instruments `fetch`, KV, D1 and Durable Object calls; custom spans use
`tracing.enterSpan()` (changelog 2026-06-16). Check the current docs for these before
building: propagation behaviour was still changing.

1. `wrangler.jsonc`:
   ```jsonc
   "observability": {
     "traces": { "enabled": true, "head_sampling_rate": 1, "destinations": ["zest-traces"] }
   }
   ```
   The destination `zest-traces` is created in the dashboard (Workers & Pages → Observability →
   Destinations): the OTLP traces endpoint plus the auth header. Its secret lives in
   Cloudflare's destination config, not in the Worker.
2. In `proxy()`: wrap the KV lookup in `enterSpan("auth")`, the DO admit in
   `enterSpan("admit")`, and the router fetch in `enterSpan("router.call")`, setting the
   attributes above and `http.response.status_code`.
3. Propagation to the router: if native tracing injects `traceparent` on the outbound
   fetch, done. If not, build it from the active span's ids (`00-<trace>-<span>-01`) and set
   it on the fetch's headers. Since the router authenticates the edge by HMAC, add
   `traceparent` to what `routerHeaders` signs (or at least never trust a client's
   `traceparent` past the edge: drop any incoming one unless a decision is made to accept
   client traces).
4. Fallback, if native export can't carry custom spans or ids: `@microlabs/otel-cf-workers`
   with `exporter: { url: env.OTEL_EXPORTER_OTLP_ENDPOINT + "/v1/traces", headers }`.
   Then the Worker needs a var `OTEL_EXPORTER_OTLP_ENDPOINT`, a secret
   `OTEL_EXPORTER_OTLP_HEADERS` (`wrangler secret put`), and `OTEL_SERVICE_NAME=zest-edge`.

Env / secrets (edge): native path — destination `zest-traces` (dashboard), nothing in the
Worker. Fallback — var `OTEL_EXPORTER_OTLP_ENDPOINT`, var `OTEL_SERVICE_NAME=zest-edge`,
secret `OTEL_EXPORTER_OTLP_HEADERS`.

### Router (Rust, axum, `router/`)

1. Deps behind a feature `otel` (default on for deploys is fine there):
   `opentelemetry 0.30`, `opentelemetry_sdk 0.30`, `opentelemetry-otlp 0.30` with
   `http-proto` + `reqwest-client` (async client: the router is already on tokio),
   `tracing-opentelemetry 0.31`. Add the OTel layer to the existing `tracing_subscriber`
   registry in `main.rs`, so `tracing` spans become OTel spans.
2. `proxy::chat`: extract the parent with `TraceContextPropagator` from the incoming
   headers (only after `identify` has verified the HMAC). Root span `chat` (kind server).
   Child spans: `admit` around `state.limits.admit`; `upstream.connect` from `send()` until
   it resolves (headers in); `upstream.ttfb` from then to the first body chunk;
   `stream` from the first chunk until `Tracker::end`, with `outcome`, `status`,
   `bytes`, and the usage tokens the tracker already parses.
3. Forward `traceparent` (and `tracestate`) to tang-llm: inject the `upstream.connect`
   span's context into the reqwest request headers.
4. Metrics (optional, same exporter): `zest_router.ttfb` histogram, `zest_router.rejected`
   counter by reason (the existing `/metrics` counters can move here).

Env (router, in `~/.config/zest/zest.env` / the systemd unit): `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_HEADERS` (secret — keep it in `secrets.env`, chmod 600),
`OTEL_SERVICE_NAME=zest-router`, optionally
`OTEL_RESOURCE_ATTRIBUTES=deployment.environment=phase-a,host.name=mew`.

### tang-llm (on the node)

Build with `--features cuda,otel`; set `TANG_OTEL_LIVE=1`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_HEADERS` (same secret as the router), `OTEL_SERVICE_NAME=tang-llm`.

### Backend values

| | Honeycomb | Grafana Cloud |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `https://api.honeycomb.io` (EU: `https://api.eu1.honeycomb.io`) | the stack's OTLP gateway, `https://otlp-gateway-<zone>.grafana.net/otlp` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `x-honeycomb-team=<ingest key>` | `Authorization=Basic <base64(instance_id:token)>` |
| Edge destination | same endpoint + `/v1/traces`, same header | same |

Sampling: keep everything at phase-A volumes; later, head-sample at the edge
(`head_sampling_rate`) and let the router and tang-llm follow the parent's sampled flag
(the SDK default `ParentBased(AlwaysOn)` does).

### Done when

One request through the public edge shows a single trace with edge, router and tang-llm
spans, `zest.request_id` = tang-llm's `request.id`, and edge `router.call` ≈ router `chat` +
network.
