#!/usr/bin/env bash
# Mirror a pinned Hugging Face revision into an S3-compatible bucket (Cloudflare R2 for frog:
# bucket `frog-weights`, served at https://weights.frog.toys), and print a catalog entry for
# it: what `POST /models/pull` takes (see crates/tang-llm/docs/node.md).
#
#   scripts/mirror-weights.sh <org/name> <revision sha> [display name] > entry.json
#
# Layout in the bucket: <org>/<name>/<revision>/<file>. Each file is downloaded from HF,
# checked against HF's LFS sha256 (when it has one), uploaded with one signed PUT and listed
# with its size and sha256. Only licences that allow redistribution are mirrored (apache-2.0,
# mit, gemma); a licence file in the repo goes along, and Gemma's is required.
#
# Env:
#   S3_ENDPOINT   https://<account>.r2.cloudflarestorage.com (or http://127.0.0.1:9000 for MinIO)
#   S3_BUCKET     frog-weights
#   S3_KEY, S3_SECRET   access key pair (R2 API token with object write)
#   S3_REGION     auto (R2) / us-east-1 (MinIO); default auto
#   PUBLIC_BASE   where the bucket is served, e.g. https://weights.frog.toys (catalog URLs)
#   HF_TOKEN      for gated repos (optional)
#   WORK          scratch directory (default: mktemp); files are deleted after upload
#   DRY_RUN=1     download and hash, don't upload
#
# Needs curl (>= 7.75, for --aws-sigv4), jq, sha256sum or shasum. Run it on a pod, not a laptop:
# it downloads the whole checkpoint. R2 takes at most 5 GiB per PUT; larger files are refused.
set -euo pipefail

repo=${1:?usage: mirror-weights.sh <org/name> <revision sha> [display name]}
rev=${2:?revision sha required (pin it: no branches)}
display=${3:-${repo#*/}}
[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a 40-hex commit sha, not $rev" >&2; exit 2; }
region=${S3_REGION:-auto}
dry=${DRY_RUN:-0}
if [[ $dry != 1 ]]; then
  : "${S3_ENDPOINT:?}" "${S3_BUCKET:?}" "${S3_KEY:?}" "${S3_SECRET:?}"
fi
: "${PUBLIC_BASE:?PUBLIC_BASE, e.g. https://weights.frog.toys}"

hf=(curl -fsSL --retry 5 --retry-delay 2)
[[ -n ${HF_TOKEN:-} ]] && hf+=(-H "Authorization: Bearer $HF_TOKEN")
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

work=${WORK:-$(mktemp -d)}
mkdir -p "$work"
trap '[[ -z ${WORK:-} ]] && rm -rf "$work"' EXIT

# Licence.
info=$("${hf[@]}" "https://huggingface.co/api/models/$repo/revision/$rev")
license=$(jq -r '.cardData.license // (.tags[]? | select(startswith("license:")) | ltrimstr("license:")) // empty' <<<"$info" | head -1)
case $license in
  apache-2.0|mit|gemma) ;;
  *) echo "$repo: licence '${license:-none}' isn't on the redistribution allowlist (apache-2.0, mit, gemma)" >&2; exit 3 ;;
esac

# Files: the checkpoint, its tokenizer and templates, and any licence; not README, images,
# other formats, or `original/` (gpt-oss's torch copy).
tree=$("${hf[@]}" "https://huggingface.co/api/models/$repo/tree/$rev?recursive=1")
files=$(jq -r '.[] | select(.type == "file") | .path' <<<"$tree" \
  | grep -E '(^|/)(LICENSE|LICENCE|NOTICE|USE_POLICY)[^/]*$|\.(json|jinja|safetensors|model|tiktoken|txt)$' \
  | grep -vE '^original/|(^|/)README|\.gitattributes$' || true)
[[ -n $files ]] || { echo "$repo@$rev: no files" >&2; exit 4; }
grep -qx config.json <<<"$files" || { echo "$repo@$rev: no config.json" >&2; exit 4; }
grep -q '\.safetensors$' <<<"$files" || { echo "$repo@$rev: no safetensors" >&2; exit 4; }
licfile=$(grep -iE '(^|/)(LICENSE|LICENCE)[^/]*$' <<<"$files" | head -1 || true)
if [[ $license == gemma && -z $licfile ]]; then
  echo "$repo: Gemma's terms require passing the licence along, and this repo has no LICENSE file" >&2; exit 3
fi

entries=()
while IFS= read -r path; do
  local_file="$work/$(tr / _ <<<"$path")"
  echo "  $path" >&2
  "${hf[@]}" -o "$local_file" "https://huggingface.co/$repo/resolve/$rev/$path"
  size=$(wc -c <"$local_file" | tr -d ' ')
  sum=$(sha256 "$local_file")
  want=$(jq -r --arg p "$path" '.[] | select(.path == $p) | .lfs.oid // empty' <<<"$tree")
  if [[ -n $want && $want != "$sum" ]]; then
    echo "$path: sha256 $sum, HF says $want" >&2; exit 5
  fi
  key="$repo/$rev/$path"
  if [[ $dry != 1 ]]; then
    (( size <= 5368709120 )) || { echo "$path: $size bytes is over R2's 5 GiB single-PUT limit" >&2; exit 6; }
    curl -fsS --retry 5 --retry-delay 2 --aws-sigv4 "aws:amz:$region:s3" --user "$S3_KEY:$S3_SECRET" \
      -H "x-amz-content-sha256: UNSIGNED-PAYLOAD" -H "x-amz-meta-sha256: $sum" \
      -T "$local_file" "$S3_ENDPOINT/$S3_BUCKET/$key" >/dev/null
  fi
  rm -f "$local_file"
  entries+=("$(jq -n --arg path "$path" --arg url "$PUBLIC_BASE/$key" --arg sha "$sum" --argjson size "$size" \
    '{path: $path, url: $url, sha256: $sha, size: $size}')")
done <<<"$files"

config=$("${hf[@]}" "https://huggingface.co/$repo/resolve/$rev/config.json")
family=$(jq -r '.model_type' <<<"$config")
printf '%s\n' "${entries[@]}" | jq -s \
  --arg id "$repo" --arg name "$display" --arg family "$family" --arg rev "$rev" --arg license "$license" \
  --arg licfile "$licfile" '{
    id: $id, display_name: $name, family: $family, revision: $rev, license: $license,
    license_file: (if $licfile == "" then null else $licfile end),
    weight_bytes: (map(select(.path | endswith(".safetensors")) | .size) | add),
    total_bytes: (map(.size) | add),
    files: .
  }'
