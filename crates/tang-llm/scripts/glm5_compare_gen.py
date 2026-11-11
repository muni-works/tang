#!/usr/bin/env python3
"""R2b greedy-parity gate: compare tang flash-generate ids against the prompt suite.

W0 harness. The suite (20 prompts) and its HF-greedy EXPECTED ids live in
crates/tang-llm/tests/fixtures/glm5/prompts/ (prompts.json + per-prompt .ids/.expected).
tang runs, per prompt:

    tang-llm flash-generate <gguf> --ids-file <prompt>.ids -n <n_new> --temp 0 \
        --out <got>.ids --quiet

and this script diffs every got/expected pair:

    python3 glm5_compare_gen.py --suite <dir> [--got <dir>]

- PASS: got ids == expected ids (expected may end early on EOS; a got run must stop at
  the same point — compare min(len) prefix and flag an EOS mismatch separately).
- FAIL prints the first-divergence index and the ids/text around it.

Also `--print-runner` emits the shell snippet that runs the whole suite through
tang-llm flash-generate (CPU ref: TANG_GLM5_GPU=0; GPU: TANG_GPUS=1 default engine path)
so the same commands can be pasted on a pod.

Exit code 0 = all pass, 1 = any fail/missing.
"""
import argparse
import json
import os
import sys


def load_suite(suite_dir):
    with open(os.path.join(suite_dir, "prompts.json")) as f:
        doc = json.load(f)
    return doc


def read_ids(path):
    if not os.path.exists(path):
        return None
    s = open(path).read()
    return [int(w) for w in s.replace(",", " ").split()]


def first_divergence(got, want):
    n = min(len(got), len(want))
    for i in range(n):
        if got[i] != want[i]:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "..", "tests", "fixtures", "glm5", "prompts"))
    ap.add_argument("--got", help="dir with <name>.ids from tang (default: --suite/got)")
    ap.add_argument("--print-runner", action="store_true",
                    help="print the tang-llm runner snippet and exit")
    ap.add_argument("--gguf", default="/private/tmp/glm5-tiny/glm5-tiny-q8_0.gguf",
                    help="GGUF path for the runner snippet")
    ap.add_argument("--env", default="",
                    help="env prefix for the runner snippet (e.g. 'TANG_GLM5_GPU=0' or 'TANG_GPUS=1')")
    a = ap.parse_args()

    doc = load_suite(a.suite)
    got_dir = a.got or os.path.join(a.suite, "got")

    if a.print_runner:
        env = f"{a.env} " if a.env else ""
        print("# R2b runner — run from the fixture dir; then glm5_compare_gen.py --suite .")
        print(f"mkdir -p {got_dir}")
        for e in doc["prompts"]:
            print(f"{env}tang-llm flash-generate {a.gguf} --ids-file {e['name']}.ids "
                  f"-n {e['n_expected']} --temp 0 --out {got_dir}/{e['name']}.ids --quiet")
        print(f"python3 glm5_compare_gen.py --suite {a.suite} --got {got_dir}")
        return

    npass = nfail = 0
    rows = []
    for e in doc["prompts"]:
        want = e["expected_ids"]
        got = read_ids(os.path.join(got_dir, e["name"] + ".ids"))
        if got is None:
            rows.append((e["name"], "MISSING", "-", len(want)))
            nfail += 1
            continue
        d = first_divergence(got, want)
        eos = doc["eos_ids"]
        # stop handling: an EOS in `want` means the HF run stopped there; tang may also
        # stop (then got ends). Compare the min-length prefix and check the stop point.
        if d is None:
            if len(got) == len(want):
                status = "PASS"
            elif len(got) < len(want):
                # tang stopped early: only OK if its last token is an EOS id and want
                # continues with the same EOS-driven termination
                status = "PASS" if (got and got[-1] in eos and len(want) > len(got)
                                    and want[len(got) - 1 if len(got) else -1] in eos) else "LEN"
                if status != "PASS":
                    nfail += 1
            else:
                status = "LONG"
                nfail += 1
            if status == "PASS":
                npass += 1
            rows.append((e["name"], status, "-", len(want)))
        else:
            nfail += 1
            ctx = 4
            lo = max(0, d - ctx)
            rows.append((e["name"], "FAIL", str(d), len(want)))
            print(f"{e['name']}: FAIL first divergence at index {d} "
                  f"(want {want[d]}, got {got[d] if d < len(got) else '<end>'}; "
                  f"want[{lo}:{d + ctx}] = {want[lo:d + ctx]}, "
                  f"got[{lo}:{d + ctx}] = {got[lo:d + ctx]})")

    for name, status, div, ln in rows:
        print(f"{name:24s} {status:8s} first-div {div:>6s} expected_len {ln}")
    print(f"pass {npass}/{npass + nfail}")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
