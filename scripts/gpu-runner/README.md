# gpu-runner

A warm runner on each H200 pod, so a GPU check takes minutes, not a fresh pod
setup, build and weight upload.

```
scripts/gpu-run.sh h200-1 install                 # copy to the pod, start (idempotent)
scripts/gpu-run.sh h200-1 my-branch check         # tiny parity + short Qwen3-8B timing
scripts/gpu-run.sh h200-1 my-branch race:qwen3-8b LENS=128,4096
scripts/gpu-run.sh h200-2 my-branch race:coder-next
scripts/gpu-run.sh h200-1 my-branch 'cmd:cargo test --release -p tang-gpu --features cuda -- --test-threads=1'
scripts/gpu-run.sh h200-1 status
```

Run `tang-gpu` and `tang-bench` test suites on a runner pod with `--test-threads=1`:
with parallel test threads `tang-bench --lib` segfaults intermittently on the pods
(not root-caused; never seen single-threaded). The pods have no Vulkan driver, so
the wgpu tests in `tang-gpu` skip (they print `skipped: no wgpu adapter`).

The client blocks, with one server-side wait that checks every 60 s, then
prints `result.json`. The branch has to be pushed to muni-works/tang.

## What stays warm

- **Build:** `~/runner/src` is a single tang clone that every job reuses. A
  job runs `git fetch` and `git checkout --detach`. Git rewrites only the
  files that changed, so cargo's fingerprints and `src/target` stay valid. A
  branch with no changes builds in seconds. The toolchain and sccache come
  from `/models/bootstrap/tang-up.sh`.
- **Weights:** `~/runner/models/<name>` holds weights pulled once to local
  disk: from the private R2 bucket `frog-models` (pinned revisions under
  `<org>/<name>/<sha>/`) when the pod has `R2_KEY` / `R2_SECRET`, else copied from
  the `/models` network volume. On a 4090 pod the 49.6 GB coder-next GGUF takes
  150–270 s from R2 (rclone, 16 streams: 0.18–0.33 GB/s) against 36 min from the volume
  (0.023 GB/s). Create runner pods with
  `R2_KEY={{ RUNPOD_SECRET_r2_access_key_id }}` and
  `R2_SECRET={{ RUNPOD_SECRET_r2_secret_access_key }}`; without them, or on any R2
  error, the runner uses the volume as before. After the first job they are also
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

Every job with a model (check, race and cmd) gets board rows from
`bench_board.py parse`, which reads `out/` or, failing that, the log. The job's
`TANG_*` env becomes the row's `config`, so a `TANG_COKERNEL=1` run doesn't
plot as the default. Rows are tagged `source: gpu-runner`, appended to
`~/runner/bench/results/board.jsonl` and returned in `result.json`.
`gpu-run.sh` then appends them to `bench/results/board.jsonl` on main and
pushes, through `scripts/board-push.sh`. That push happens on the machine that
ran `gpu-run.sh`, because the pods hold no GitHub credentials. Set
`BOARD_PUSH=0` to skip it. `install` copies `bench_board.py` next to
`runner.py`.

Several sessions share each pod, and other GPU work (in `~/F` and the other
session dirs) is not queued here. The runner serializes only its own jobs.
