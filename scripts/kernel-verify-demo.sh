#!/usr/bin/env bash
# The tang-verify planted-bug demo: plant the GLM-5.3 KDA k/v byte-offset bug (the real
# bug fixed in cc578187), run the real model once with the harness on, and print the
# one-screen verdict, then the same run with the fix.
#
#   scripts/kernel-verify-demo.sh            # build both binaries, run both, print
#   scripts/kernel-verify-demo.sh --replay   # print the verdict from the last run's logs
#
# Needs a CUDA box with the GLM-5.3-Flash GGUF (KDA resident fits a 24 GB 3090 with the
# FFNs on the CPU). Env: MODEL (first .gguf shard), OUT (scratch dir, default
# ./target/kernel-verify-demo), LOCK (GPU lock file; used with flock when it exists).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=${OUT:-$ROOT/target/kernel-verify-demo}
MODEL=${MODEL:-$(find "$HOME/tang-scratch/glm53/UD-Q4_K_XL" -name '*-00001-of-*.gguf' 2>/dev/null | head -1)}
LOCK=${LOCK:-$HOME/.cache/tang-gpu.lock}
PROMPT="Write a haiku about the sea."
mkdir -p "$OUT"

b=$'\e[1m'; d=$'\e[2m'; r=$'\e[31m'; g=$'\e[32m'; y=$'\e[33m'; n=$'\e[0m'
[ -t 1 ] || { b=; d=; r=; g=; y=; n=; }

KDA=crates/tang-llm/src/flash/glm5/kda_dev.rs
GOOD='(&self.wk, di as u64 * f), (&self.wv, 2 * di as u64 * f)'
BAD='(&self.wk, di as u64), (&self.wv, 2 * di as u64)'

build() { # build <tree> <binary out>
    ( cd "$1" && export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda \
        CARGO_TARGET_DIR=$OUT/target && touch crates/tang-llm/src/lib.rs &&
      cargo build --release -p tang-llm --bin tang-llm --features cuda ) > "$2.build.log" 2>&1 ||
        { echo "build failed: $2.build.log"; tail -20 "$2.build.log"; exit 1; }
    grep -q "Compiling tang-llm" "$2.build.log" || { echo "no recompile of tang-llm — refusing a stale binary"; exit 1; }
    cp "$OUT/target/release/tang-llm" "$2"
}

run() { # run <binary> <log>
    local pre=()
    [ -e "$LOCK" ] && pre=(flock "$LOCK")
    # KDA on the GPU (FFNs on the CPU so it fits 24 GB), the KDA shadow check on, one
    # token (the 20-token prefill window is the check)
    TANG_GLM5_GPU_FFN=0 TANG_GLM5_KDA_MAXT=64 TANG_GLM5_KDA_DEBUG=1 \
        "${pre[@]}" "$1" flash-generate "$MODEL" --prompt "$PROMPT" -n 1 --temp 0 > "$2" 2>&1 || true
}

if [ "${1:-}" != "--replay" ]; then
    [ -n "$MODEL" ] && [ -e "$MODEL" ] || { echo "set MODEL to the first GLM-5.3 .gguf shard"; exit 1; }
    grep -qF "$GOOD" "$KDA" || { echo "$KDA: the fixed launch line isn't where the demo expects it"; exit 1; }
    echo "${d}planting the bug in a scratch copy, building both binaries (a few minutes)...${n}"
    rsync -rlp --checksum --delete --exclude target --exclude .git --exclude .frog "$ROOT/" "$OUT/planted/"
    cp "$OUT/planted/$KDA" "$OUT/planted/$KDA.orig"
    src=$(cat "$OUT/planted/$KDA.orig")
    printf '%s\n' "${src/"$GOOD"/"$BAD"}" > "$OUT/planted/$KDA"
    grep -qF "$BAD" "$OUT/planted/$KDA" || { echo "plant failed"; exit 1; }
    diff -u "$OUT/planted/$KDA.orig" "$OUT/planted/$KDA" > "$OUT/planted.diff" || true
    rm "$OUT/planted/$KDA.orig"
    build "$OUT/planted" "$OUT/tang-llm-planted"
    build "$ROOT" "$OUT/tang-llm-fixed"
    echo "${d}running the real model twice (planted, fixed)...${n}"
    run "$OUT/tang-llm-planted" "$OUT/planted.log"
    run "$OUT/tang-llm-fixed" "$OUT/fixed.log"
fi

P=$OUT/planted.log F=$OUT/fixed.log
[ -e "$P" ] && [ -e "$F" ] || { echo "no logs in $OUT (run without --replay first)"; exit 1; }
row() { grep -m1 -E "^kdadbg w0 l$L t[0-9]+ $1 " "$P"; }
field() { sed -nE "s/.* $1 ([0-9]+)\/([0-9]+).*/\\$2/p"; }
L=$(grep -m1 -oE '^kdadbg w0 l[0-9]+' "$P" | grep -oE '[0-9]+$')
T=$(row qkv.q | grep -oE ' t[0-9]+ ' | tr -dc 0-9)
N=$(row qkv.k | sed -nE 's/.* bad [0-9]+\/([0-9]+).*/\1/p')
DI=$((N / T))
nan() { row "$1" | sed -nE 's/.* nonfinite ([0-9]+)\/.*/\1/p'; }
verdict=$(grep -m1 'kdadbg FIRST DIVERGENCE' "$P" | sed 's/^kdadbg FIRST DIVERGENCE //')

echo
echo "${b}━━ tang-verify: a planted GPU bug, caught in one run ━━${n}"
echo
echo "${b}1. The bug we planted${n} ${d}($KDA, the q/k/v projection launch)${n}"
grep -E '^[-+] ' "$OUT/planted.diff" | sed -e "s/^-/  ${g}fixed:  ${n}/" -e "s/^+/  ${r}planted:${n}/" | sed 's/  *for (w, off) in / /'
echo "  A GPU pointer counts BYTES; the planted line adds a count of FLOATS (4 bytes each)."
echo "  So k is written 1/4 of the way, and v 1/2, into the q block. This exact bug shipped"
echo "  in the GLM-5.3 engine on Oct 26 and was found by this method (fix: cc578187)."
echo
echo "${b}2. One run of the real model with the harness on${n} ${d}(GLM-5.3, real weights, layer $L, first window: $T tokens)${n}"
echo "  Every output buffer is filled with NaN first, so memory no kernel wrote stays NaN."
for t in qkv.q qkv.k qkv.v; do
    k=$(nan $t); flag="${g}ok${n}"; row $t | grep -q ' BAD ' && flag="${r}WRONG${n}"
    printf "    %-6s %b   %7s of %s values never written\n" "$t" "$flag" "$k" "$N"
done
EK=$((T * DI / 2)) EV=$((T * DI))
ok() { [ "$1" = "$2" ] && echo "${g}✓${n}" || echo "${r}✗ (got $1)${n}"; }
echo "  Predicted by the offset arithmetic ($T tokens × $DI floats per section):"
echo "    k: v's misplaced writes stop halfway through k → $T × $DI/2 = $EK unwritten  $(ok "$(nan qkv.k)" $EK)"
echo "    v: nothing writes v's block at all             → $T × $DI   = $EV unwritten  $(ok "$(nan qkv.v)" $EV)"
echo "  ${b}Harness verdict:${n}"
echo "$verdict" | sed -E 's/^w[0-9]+ l[0-9]+ t[0-9]+: //; s/ \(writer/\n      (writer/; s/: expected =/:\n       expected =/; s/ → suspect:/\n    → suspect:/' | sed "s/^/    ${y}/; s/$/${n}/"
echo "  ${d}(the naive rule, \"first failing stage\", would have blamed attn_q: its block is the${n}"
echo "  ${d} first to fail, but the values in it are k's)${n}"
echo
echo "${b}3. The same run with the fix${n}"
grep -m1 'kdadbg SUMMARY:' "$F" | sed -E "s/^kdadbg /  ${g}/; s/$/${n}/"
grep -m1 -E '^ids:' "$F" | sed 's/^/  /' || true
echo
echo "${d}logs: $P, $F${n}"
