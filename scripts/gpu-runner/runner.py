#!/usr/bin/env python3
"""Warm GPU runner: one per pod, a spool of jobs in two lanes, one GPU job at a time.

    python3 runner.py serve            # the daemon (start.sh runs it under nohup)
    python3 runner.py submit JOB.json  # copy a job into the spool, print its id
    python3 runner.py status           # queue, running job, last results
    python3 runner.py status --json    # the same for gpu-run.sh's pod pick (auto)

Layout (RUNNER_HOME, default ~/runner):
    src/            one tang clone, reused by every job: `git checkout` touches only
                    the files that changed, so cargo's fingerprints and src/target stay
                    hot and an unchanged branch builds in seconds
    models/<name>/  weights copied once from /models (the network volume) to local
                    disk; after the first job they also sit in the page cache
    jobs/<id>.json  pending jobs: short lane first, then long, FIFO by mtime within a lane
                    (a running job is never interrupted: GPU timings stay clean)
    jobs/<id>/      job.json, log.txt, result.json, DONE (written last)
    bench/results/board.jsonl   board rows (scripts/bench_board.py format)

Job file (all but "job" optional):
    {"job": "check" | "race" | "cmd",
     "ref": "main",              branch, tag or commit of muni-works/tang
     "model": "qwen3-8b",        qwen3-8b | coder-next | ds  (sets MODEL_DIR)
     "script": "scripts/derived-race.sh",   race: script in the checkout
     "cmd": "cargo run ...",     cmd: shell line in the checkout
     "env": {"LENS": "128"},     extra env for the job
     "timeout_s": 7200,
     "lane": "short"}            short (smokes, checks: minutes) | long (default for race/cmd)
check = build qwen_decode, then check.sh (tiny parity + short Qwen3-8B timing).
race  = the race script with MODEL_DIR and OUT set; it builds what it needs.
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

HOME = os.path.expanduser(os.environ.get("RUNNER_HOME", "~/runner"))
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HOME, "src")
JOBS = os.path.join(HOME, "jobs")
MODELS = os.path.join(HOME, "models")
BOARD = os.path.join(HOME, "bench/results/board.jsonl")
REPO_URL = os.environ.get("RUNNER_REPO", "https://github.com/muni-works/tang")
VOLUME = {"qwen3-8b": "/models/qwen3-8b-4bit", "coder-next": "/models/coder-next",
          "ds": "/models/ds41-flash"}
# models too big to copy to the 200 GB container disk are read from the volume
NO_COPY = {"ds"}
ENV = os.path.expanduser("~/.tang-up/env")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd, log, cwd=None, env=None, timeout=None):
    """bash with the tang-up env; output appended to log. Returns (rc, seconds)."""
    t = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        pre = f"[ -f {ENV} ] && . {ENV}; export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda}}; "
        try:
            rc = subprocess.run(["bash", "-c", pre + cmd], cwd=cwd, env=env, stdout=f,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[runner] timeout after {timeout}s\n")
            rc = 124
    return rc, round(time.time() - t, 1)


def checkout(ref, log):
    if not os.path.isdir(os.path.join(SRC, ".git")):
        rc, _ = sh(f"git clone -q {REPO_URL} {SRC}", log)
        if rc:
            return None
    # a branch, tag or full SHA fetches directly; a short SHA can't be
    # fetched by name, so fetch all branches and resolve it locally
    rc, _ = sh(f"git fetch -q --no-prune origin {ref} && git checkout -q -f --detach FETCH_HEAD",
               log, cwd=SRC)
    if rc:
        rc, _ = sh("git fetch -q --no-prune origin '+refs/heads/*:refs/remotes/origin/*'"
                   f" && git checkout -q -f --detach \"$(git rev-parse --verify -q {ref}^{{commit}})\"",
                   log, cwd=SRC)
    if not rc:
        rc, _ = sh("git clean -fdq -e target", log, cwd=SRC)
    if rc:
        return None
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SRC,
                          capture_output=True, text=True).stdout.strip()


def log_line(log, msg):
    with open(log, "a") as f:
        f.write(msg + "\n")


def copy_ranges(src, dst, parts=16, chunk=8 << 20):
    """Copy one file in `parts` parallel byte ranges (the object-backed volume serves
    ~45 MB/s a stream; ranges in parallel go several times faster)."""
    import concurrent.futures
    n = os.path.getsize(src)
    with open(dst, "wb") as f:
        f.truncate(n)
    step = -(-n // parts) if n else 0

    def one(i):
        lo, hi = i * step, min(n, (i + 1) * step)
        with open(src, "rb", buffering=0) as a, open(dst, "r+b", buffering=0) as b:
            a.seek(lo)
            b.seek(lo)
            while lo < hi:
                # the volume's FUSE mount can fail a read with ENOMEM under load:
                # retry that chunk (a whole-file restart costs ~20 min on 50 GB)
                for tries in range(6):
                    try:
                        a.seek(lo)
                        buf = a.read(min(chunk, hi - lo))
                        break
                    except OSError:
                        if tries == 5:
                            raise
                        time.sleep(2 << tries)
                if not buf:
                    raise IOError(f"short read in {src} at {lo}")
                b.write(buf)
                lo += len(buf)

    with concurrent.futures.ThreadPoolExecutor(parts) as ex:
        list(ex.map(one, range(parts)))


def model_dir(name, log):
    """The model's local copy (~/runner/models/<name>): copied from the volume once, in
    parallel byte ranges, marked .runner-complete and reused by every later job. Read from
    the volume in place only when the local disk can't hold it (or the copy fails)."""
    if not name:
        return None
    vol = VOLUME.get(name, name)
    if name in NO_COPY or not os.path.isdir(vol):
        return vol
    local = os.path.join(MODELS, name)
    done = os.path.join(local, ".runner-complete")
    if os.path.exists(done):
        return local
    files = [(r, f) for r, _, fs in os.walk(vol) for f in fs]
    need = sum(os.path.getsize(os.path.join(r, f)) for r, f in files)
    os.makedirs(local, exist_ok=True)
    have = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local) for f in fs)
    free = shutil.disk_usage(local).free
    headroom = float(os.environ.get("RUNNER_DISK_HEADROOM_GB", "20")) * 1e9
    if need - have + headroom > free:
        log_line(log, f"[runner] model {name}: {need / 1e9:.1f} GB won't fit "
                      f"({free / 1e9:.1f} GB free); reading {vol} in place")
        return vol
    t = time.time()
    try:
        for r, f in files:
            src = os.path.join(r, f)
            dst = os.path.join(local, os.path.relpath(src, vol))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
                    and f != ".complete":
                continue  # a file a previous (interrupted) copy finished
            copy_ranges(src, dst + ".part")
            os.replace(dst + ".part", dst)
    except Exception as e:
        log_line(log, f"[runner] model {name}: copy failed ({e!r}); reading {vol} in place")
        return vol
    open(done, "w").write(now() + "\n")
    dt = time.time() - t
    log_line(log, f"[runner] model {name}: copied {need / 1e9:.1f} GB to {local} in {dt:.0f} s "
                  f"({need / 1e6 / max(dt, 1e-3):.0f} MB/s)")
    return local


def gpu_name():
    """This pod's GPU, e.g. "NVIDIA H200"; rows whose output doesn't name one get it."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return p.stdout.splitlines()[0].strip() if p.returncode == 0 and p.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def board_rows(out, model, ref, log, env):
    """Board rows from the job's OUT dir (else its log), labeled with the job's
    TANG_* env; appended to the local board and returned in result.json, from
    which gpu-run.sh pushes them to main (the pod holds no credentials)."""
    parser = os.path.join(HERE, "..", "bench_board.py")
    if not os.path.exists(parser):  # installed copy: bench_board.py beside runner.py
        parser = os.path.join(HERE, "bench_board.py")
    config = " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("TANG_"))
    p, gpu = None, gpu_name()
    for src in (out, log):
        cmd = [sys.executable, parser, "parse", src, "--model", model, "--branch", ref,
               "--source", "gpu-runner"] + (["--config", config] if config else []) \
            + (["--gpu", gpu] if gpu else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            break
    if p.returncode or not p.stdout.strip():
        with open(log, "a") as f:
            f.write(f"\n[runner] no board rows: {p.stderr.strip()[-500:]}\n")
        return []
    os.makedirs(os.path.dirname(BOARD), exist_ok=True)
    with open(BOARD, "a") as f:
        f.write(p.stdout if p.stdout.endswith("\n") else p.stdout + "\n")
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def run(job_id, job, queued_at):
    d = os.path.join(JOBS, job_id)
    log = os.path.join(d, "log.txt")
    out = os.path.join(d, "out")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    res = {"id": job_id, "job": job.get("job"), "ref": job.get("ref", "main"),
           "model": job.get("model"), "started": now(), "queue_s": round(t0 - queued_at, 1)}
    kind = job.get("job", "check")
    timeout = int(job.get("timeout_s", 7200))

    t = time.time()
    commit = checkout(res["ref"], log)
    res["fetch_s"] = round(time.time() - t, 1)
    res["commit"] = commit
    if not commit:
        return dict(res, ok=False, error="checkout failed", total_s=round(time.time() - t0, 1))

    model = job.get("model") or ("qwen3-8b" if kind == "check" else None)
    res["model"] = model
    t = time.time()
    mdir = model_dir(model, log)
    res["model_s"] = round(time.time() - t, 1)
    env = dict(os.environ, OUT=out, **{k: str(v) for k, v in job.get("env", {}).items()})
    if mdir:
        env["MODEL_DIR"] = mdir

    if kind == "check":
        rc, res["build_s"] = sh("cargo build --release -p tang-bench --features cuda "
                                "--example qwen_decode", log, cwd=SRC, env=env, timeout=timeout)
        if rc:
            return dict(res, ok=False, error="build failed", total_s=round(time.time() - t0, 1))
        rc, res["run_s"] = sh(f"bash {HERE}/check.sh", log, cwd=SRC, env=env, timeout=timeout)
    elif kind == "race":
        script = job.get("script") or "scripts/derived-race.sh"
        rc, res["run_s"] = sh(f"bash {script}", log, cwd=SRC, env=env, timeout=timeout)
    elif kind == "cmd":
        rc, res["run_s"] = sh(job["cmd"], log, cwd=SRC, env=env, timeout=timeout)
    else:
        return dict(res, ok=False, error=f"unknown job {kind!r}")

    res["exit"] = rc
    res["ok"] = rc == 0
    if model:  # every job with a model: check, race and cmd
        res["board"] = board_rows(out, model, res["ref"], log, job.get("env", {}))
    summ = os.path.join(out, "summary.txt")
    if os.path.exists(summ):
        res["summary"] = open(summ, errors="replace").read()[-4000:]
    res["total_s"] = round(time.time() - t0, 1)
    return res


def lane(job):
    """short | long: a check is short unless it says otherwise; anything else long."""
    lane = job.get("lane") or ("short" if job.get("job", "check") == "check" else "long")
    return "short" if lane == "short" else "long"


def pending():
    """Pending job names in run order: the short lane, then the long, FIFO in each."""
    try:
        names = [n for n in os.listdir(JOBS) if n.endswith(".json")]
    except FileNotFoundError:
        return []

    def key(n):
        p = os.path.join(JOBS, n)
        try:
            lane_n = lane(json.load(open(p)))
        except Exception:
            lane_n = "short"  # a bad file fails fast
        return (lane_n != "short", os.path.getmtime(p))

    return sorted(names, key=key)


def next_job():
    names = pending()
    return names[0] if names else None


def compiles(path):
    """`path` parses as Python (a half-copied runner.py doesn't take over)."""
    try:
        compile(open(path).read(), path, "exec")
        return True
    except (SyntaxError, ValueError, OSError):
        return False


def serve():
    os.makedirs(JOBS, exist_ok=True)
    with open(os.path.join(HOME, "runner.pid"), "w") as f:
        f.write(str(os.getpid()))
    print(f"[runner] {now()} serving {JOBS}", flush=True)
    me = os.path.abspath(__file__)
    born = os.path.getmtime(me)
    while True:
        # a reinstalled runner.py takes over between jobs (never under a running one)
        if os.path.getmtime(me) != born and compiles(me):
            print(f"[runner] {now()} runner.py changed: re-exec", flush=True)
            os.execv(sys.executable, [sys.executable, me, "serve"])
        name = next_job()
        if not name:
            time.sleep(2)
            continue
        job_id = name[:-5]
        path = os.path.join(JOBS, name)
        queued_at = os.path.getmtime(path)
        d = os.path.join(JOBS, job_id)
        os.makedirs(d, exist_ok=True)
        try:
            job = json.load(open(path))
        except Exception as e:  # a bad file is a failed job, not a dead runner
            job = {}
            res = {"id": job_id, "ok": False, "error": f"bad job file: {e}"}
        shutil.move(path, os.path.join(d, "job.json"))
        open(os.path.join(HOME, "running"), "w").write(job_id)
        print(f"[runner] {now()} start {job_id} {json.dumps(job)}", flush=True)
        if job:
            try:
                res = run(job_id, job, queued_at)
            except Exception as e:
                res = {"id": job_id, "ok": False, "error": f"runner error: {e!r}"}
        res["finished"] = now()
        json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
        open(os.path.join(d, "DONE"), "w").write("ok\n" if res.get("ok") else "fail\n")
        os.remove(os.path.join(HOME, "running"))
        print(f"[runner] {now()} done {job_id} ok={res.get('ok')} total={res.get('total_s')}s",
              flush=True)


def submit(path):
    job = json.load(open(path)) if path != "-" else json.load(sys.stdin)
    job_id = job.pop("id", None) or (datetime.datetime.now(datetime.timezone.utc).strftime("%m%d-%H%M%S-")
                                     + uuid.uuid4().hex[:6])
    os.makedirs(JOBS, exist_ok=True)
    tmp = os.path.join(JOBS, f".{job_id}.tmp")
    json.dump(job, open(tmp, "w"))
    os.rename(tmp, os.path.join(JOBS, job_id + ".json"))
    print(job_id)


def job_lane(path):
    try:
        return lane(json.load(open(path)))
    except Exception:
        return "short"


def status():
    rp = os.path.join(HOME, "running")
    running = open(rp).read().strip() if os.path.exists(rp) else None
    queue = pending()
    if "--json" in sys.argv:
        r = None
        if running:
            r = {"id": running, "lane": job_lane(os.path.join(JOBS, running, "job.json")),
                 "for_s": round(time.time() - os.path.getmtime(rp))}
        print(json.dumps({"running": r, "pending": [
            {"id": n[:-5], "lane": job_lane(os.path.join(JOBS, n))} for n in queue]}))
        return
    print(f"running: {running}\npending: {[n[:-5] for n in queue]}")
    done = sorted((n for n in os.listdir(JOBS) if os.path.exists(os.path.join(JOBS, n, "DONE"))),
                  key=lambda n: os.path.getmtime(os.path.join(JOBS, n, "DONE")))[-5:]
    for n in done:
        r = json.load(open(os.path.join(JOBS, n, "result.json")))
        print(f"  {n}: ok={r.get('ok')} {r.get('job')} {r.get('ref')}@{r.get('commit')} "
              f"total {r.get('total_s')}s (build {r.get('build_s')}s, run {r.get('run_s')}s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"serve": serve, "status": status}.get(cmd, lambda: submit(sys.argv[2]))()
