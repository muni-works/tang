# Recursive MTP training for the Coder engine

`flash-mtp-train` (see `flash-mtp-train.md`) also trains the Qwen3-Next MTP sidecar the Coder
engine (`qwen3next`, Qwen3-Coder-Next) drafts with (`scripts/qwen3next_mtp_sidecar.py`). The
trainer detects the sidecar by `mtp.fc.weight`; everything else is the same pipeline:
residual dumps, recursive self-conditioned cell, same-size GGUF patch export.

The cell, as the engine runs it (`enqueue_mtp`):

```text
r = fc([rmsnorm(embed(t[p+1]))·enorm | rmsnorm(h_p)·hnorm])          (cx_mtp_in + mtp_fc)
r += attn_output(sigmoid(gate) · attn(q/k rmsnorm, NeoX rope on 64 dims, at position p+d))
r += Σ top-10 softmax-renormalised experts + sigmoid(gate_inp_shexp·x) · shared expert
h' = rmsnorm(r)·norm  → main head; h' is the next depth's h                    (chain on s.x)
```

`h_p` is `m.mh`: the trunk's final hidden row after the output norm. Trainable: fc, the two
input norms, attention (q|gate, k, v, o, q/k norms), both layer norms, `mtp.norm`, router,
shared expert and its gate (17 tensors, ~40M parameters). Routed experts (Q8_0, as served),
the trunk's embedding and head stay frozen. The loss is the recursive teacher-forced CE at
depths 1..`--depth` (default 3; 4 works), weighted `β^(d-1)` (`--beta`, default 0.8).

## Build (pod)

```sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_ROOT=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64
cargo build --release -p tang-llm --features cuda,mtp-train
B=target/release/tang-llm
TRUNK=/workspace/Qwen3-Coder-Next-UD-Q4_K_XL.gguf     # first shard
MTP=/workspace/mtp.gguf                               # qwen3next_mtp_sidecar.py output
```

## 1. Dump

`prompts.jsonl`: one prompt per line (a JSON string or `{"prompt": ...}`), 200 code prompts.

```sh
$B flash-coder-dump $TRUNK --mtp $MTP --prompts prompts.jsonl --out /workspace/cmtp \
  -n 2000 --k 3 --max-ctx 8192 2> dump.log
```

One load; per prompt `i`, greedy with the chat template, drafting with the sidecar's 3-step
chain (generation is exact either way). `cmtp/data/pNNN` holds every generated row
(`h.f16`, `ids.u32`), the engine's own drafts (`draft.u32`: position, depth, argmax,
probability; depth 1 on every row, deeper along each window's chain), `prompt.txt` and
`gen.txt`. `cmtp/prefix/pNNN` holds every prompt row for `--prefix-data`. Data stops at the
first `<|im_end|>`/`<|endoftext|>` (`--stop`). Completed prompts are skipped, so rerun the
same command after an interruption. `pNNN % 10 == 0` (20 of 200) are held out by the trainer.

## 2. Forward oracle (before any training)

```sh
$B flash-mtp-train $TRUNK $MTP --data /workspace/cmtp/data --prefix-data /workspace/cmtp/prefix \
  --out /workspace/cmtp/oracle --check-forward --seq 32
```

Runs the trainer cell on the first 4 windows of 4 held-out prompts with full teacher K/V and
compares each row's argmax (and probability) with the engine's draft at depth 1, and at
depths 2..3 where the engine's earlier drafts equal the teacher tokens. PASS needs ≥ 97%
depth-1 and ≥ 90% deeper agreement (int8 activations and bf16 K/V move near-ties; a port
error moves most rows). Creates no output. Do not train if it fails.

## 3. Train (pilot)

```sh
$B flash-mtp-train $TRUNK $MTP --data /workspace/cmtp/data --prefix-data /workspace/cmtp/prefix \
  --out /workspace/cmtp/run1 --steps 500 --seq 32 --burn 8 --depth 3 --beta .8 \
  --lr 1e-5 --clip 1 --eval-every 100 2> train.log
```

`--prefix-data` gives every window its full earlier teacher K/V (as serving has); without it
windows start cold. Each `step-NNNNNN/` has `mtp.gguf` (a same-size patch of `$MTP`: Q8_0
matrices requantized, F32 norms, untouched bytes identical) and resumable state
(`--resume`, new `--out`). `metrics.jsonl` has train CE/hits per depth and fixed held-out
windows. The routed-expert cache holds all 512 experts dequantized (~6 GB host RAM).

## 4. Evaluate

Measured draft acceptance on the 20 held-out prompts, fixed 3-step chain, same settings for
both files:

```sh
for f in $MTP /workspace/cmtp/run1/step-000500/mtp.gguf; do
  $B flash-coder-dump $TRUNK --mtp $f --prompts prompts.jsonl --eval --held-out \
    -n 2000 --k 3 > eval-$(basename $(dirname $f)).json
done
```

The last stdout line is JSON: per depth `proposed`, `accepted`, `conditional`
(P(accept d | accepted 1..d-1)) and `cumulative` (per window), plus tokens/window. Then
check exactness and speed on the serving path with the chosen file:

```sh
$B flash-spec-test $TRUNK --mtp /workspace/cmtp/run1/step-000500/mtp.gguf --prompt-file P -n 256
$B flash-bench $TRUNK --mtp /workspace/cmtp/run1/step-000500/mtp.gguf --draft mtp --prompt-file P -n 512
```

Lower held-out CE alone is not a reason to switch sidecars; acceptance and tok/s are.

## Tests

`cargo test -p tang-llm --features mtp-train --lib coder` (CPU): the cell against a plain loop
of the engine's chain on a synthetic trunk + sidecar (with/without prefix, depth 3 and 4),
gradients on all trainable tensors, byte-identical untouched export, the engine's sidecar
contract (`coder_mtp::check_sidecar`, also run by `Coder::load`) on a changed export and an
equal forward after reload; the dump writer's rows through the trainer's loader.

## Pilot 2: recipe gap analysis

Pilot 1 (500 steps, held-out live chain k=3): d1 78.2→82.3%, d2-cum 58.0→64.6%, d3-cum
40.0→48.3%, 2.76→2.95 tokens/window; held-out CE flat by step 300. Against the literature:

| piece | source | pilot 1 |
| --- | --- | --- |
| one shared head applied recursively, depth 3 | FastMTP | **already** (`forward_full`: one cell, `--depth`) |
| depth ≥ 2 fed the head's own output hidden state | FastMTP | **already** (`residual = out.r`, the normed `h'` the engine chains) |
| trained on the target's self-generated text | FastMTP | **already** (`flash-coder-dump` generates greedily with the target; FastMTP samples at T 0.6, greedy matches our greedy serving) |
| multi-step loss, `β^(k-1)` decay | FastMTP (β 0.6) | **already**, β 0.8 default; `--beta .6` is the paper's |
| verification-aware weights (full weight up to and including the first simulated rejection, decay restarts after) | VAT | **new**: `--vat` |
| soft target: CE/KL to the target's own next-token distribution | Soft-MTP, VAT ℓ^soft | **new**: `--soft W` |

VAT note: the paper's main method does not zero positions after the first rejection; it
restarts the decay there (`ŵ_k = 1` for `k < k*`, `w_{k-k*+1}` after). A hard cutoff is its
weaker ablation. `--vat` implements the main method, with `w = β^(k-1)`; the simulated
rejection is the head's current argmax vs the teacher token per window.

Soft targets need no re-dump: `h` in the dump is the trunk's normed final row, so the target's
distribution is `softmax(output · h)` with the frozen head the trainer already holds
(`model::teacher_logits`; depth d, row r reads residual row r+d+1). The soft term's gradient is
`W·(p − q)` added to the hard CE's `p − onehot`. `metrics.jsonl` adds per-depth `kl`,
`accepted` (windows with argmax = teacher at every depth ≤ d) and `target_agree` (rows whose
`argmax(output · h)` equals the dumped token; it should be ~all rows, Q4 trunk vs f16 `h`).

Pilot-2 command: as pilot 1 plus `--beta .6 --vat --soft 1`, from the borrowed head.

### Pilot 2 result (H200, shared GPU; 500 steps from the borrowed head, `--beta .6 --vat --soft 1`)

Held-out live chain, k=3, 20 prompts × 2000 tokens:

| head | d1 | d2-cum | d3-cum | tokens/window |
| --- | --- | --- | --- | --- |
| borrowed | 78.2% | 58.0% | 40.0% | 2.76 |
| pilot 1 (β .8, hard CE) | 82.3% | 64.6% | 48.3% | 2.95 |
| pilot 2 (β .6, VAT, soft) | 82.2% | 64.5% | 48.2% | 2.95 |

Train 88 min, eval 22 min. Identical within noise; the held-out curves (CE, teacher hits) track
pilot 1 step for step. The recipe pieces that were missing (VAT weights, soft targets) do not
move this head; the pieces that matter in FastMTP (shared recursive head, self-conditioning,
self-generated data, decayed multi-step loss) were already there. What differs from the papers
is scale: 500 steps × one 24-row window is ~12k loss rows, ~3% of one pass over the 180-prompt
dump (FastMTP: 3 epochs over 389k samples, batch 64, lr 5e-5). Next lever: batch (many windows
per step) and lr/steps, then more prompts — not the loss.
