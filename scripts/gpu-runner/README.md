# gpu-runner

A warm runner on each H200 pod, so a GPU check takes minutes, not a fresh pod
setup, build and weight upload.

```
scripts/gpu-run.sh h200-1 install                 # copy to the pod, start (idempotent)
scripts/gpu-run.sh h200-1 my-branch check         # tiny parity + short Qwen3-8B timing
scripts/gpu-run.sh h200-1 my-branch race:qwen3-8b LENS=128,4096
scripts/gpu-run.sh h200-2 my-branch race:coder-next
scripts/gpu-run.sh h200-1 my-branch 'cmd:cargo test --release -p tang-gpu --features cuda'
scripts/gpu-run.sh h200-1 status
```

The client blocks, with one server-side wait that checks every 60 s, then
prints `result.json`. The branch has to be pushed to muni-works/tang.

## What stays warm

- **Build:** `~/runner/src` is a single tang clone that every job reuses. A
  job runs `git fetch` and `git checkout --detach`. Git rewrites only the
  files that changed, so cargo's fingerprints and `src/target` stay valid. A
  branch with no changes builds in seconds. The toolchain and sccache come
  from `/models/bootstrap/tang-up.sh`.
- **Weights:** `~/runner/models/<name>` holds weights copied once from the
  `/models` network volume to local disk. After the first job they are also
  in the page cache (2 TB RAM). DS weights are read straight from the volume,
  since they are too big for the 200 GB disk.
- **Not yet: weights resident in VRAM across jobs.** Each job is a new
  process (an example binary or a race script), so device buffers die with
  it. Keeping them would take a long-lived tang process that serves jobs,
  which needs engine work. That is a follow-up.

## Jobs

The spool is `~/runner/jobs/<id>.json`, run FIFO, one at a time. Each job
writes these files to `~/runner/jobs/<id>/`:

- `job.json`
- `log.txt`
- `out/`, which holds the race or check outputs
- `result.json`, with timings `fetch_s`, `model_s`, `build_s`, `run_s` and
  `total_s`, `ok`, `commit`, `summary` and board rows
- `DONE`, written last

Check and race jobs append rows to `~/runner/bench/results/board.jsonl`. The
rows come from `bench_board.py parse` (from #225) and are tagged
`source: gpu-runner`.

Several sessions share each pod, and other GPU work (in `~/F` and the other
session dirs) is not queued here. The runner serializes only its own jobs.
