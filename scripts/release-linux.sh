#!/usr/bin/env bash
# Build Linux tang-llm release tarballs, the ones frog's node installer (frog.toys/node) downloads:
#
#   tang-llm-<arch>-unknown-linux-gnu.tar.gz          CPU, default features
#   tang-llm-<arch>-unknown-linux-gnu-cuda12.tar.gz   --no-default-features --features cuda (CUDA 12.8)
#
# each holding one file, `tang-llm`, at its root, with `<asset>.sha256` (sha256sum output) beside it.
# <arch> is this machine's (x86_64 or aarch64); there is no cross-compiling.
#
# Same build as .github/workflows/linux-release.yml, for when GitHub runners aren't available:
# on any Linux box with docker (mew), or natively on a RunPod pod (after scripts/pod-bootstrap.sh).
#
#   scripts/release-linux.sh                         # both variants, in docker, into ./dist
#   scripts/release-linux.sh --variant cpu --upload v0.4.0
#   scripts/release-linux.sh --native --variant cuda # on a pod: host toolchain + CUDA
#
# Options:
#   --variant cpu|cuda   build only one (repeatable; default both)
#   --native             build with the host's cargo (and nvcc/CUDA for cuda) instead of docker.
#                        The binary then needs the host's glibc or newer; docker builds on
#                        ubuntu 22.04 (glibc 2.35), like the workflow.
#   --upload TAG         `gh release upload TAG ... --clobber`, creating the release if needed
#   --out DIR            where tarballs go (default ./dist)
# Env:
#   CARGO_TARGET_DIR     target dir (default ~/build/target-release-<variant>; docker mounts it)
#   RELEASE_CACHE        cargo/rustup cache for docker builds (default ~/build/release-cache)
set -euo pipefail

CPU_IMAGE=ubuntu:22.04
CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04

variants=() native=0 tag="" out=dist
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) variants+=("$2"); shift ;;
    --native) native=1 ;;
    --upload) tag="$2"; shift ;;
    --out) out="$2"; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "release-linux: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ ${#variants[@]} -gt 0 ] || variants=(cpu cuda)
[ "$(uname -s)" = Linux ] || { echo "release-linux: run this on Linux" >&2; exit 1; }

case "$(uname -m)" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) echo "release-linux: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
triple="$arch-unknown-linux-gnu"
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)

cargo_args() {
  case "$1" in
    cpu) echo "build --release -p tang-llm --target $triple" ;;
    cuda) echo "build --release -p tang-llm --no-default-features --features cuda --target $triple" ;;
    *) echo "release-linux: unknown variant $1" >&2; exit 2 ;;
  esac
}

# Runs inside the container: apt deps + rustup into the mounted cache, then cargo.
in_container='set -eu
export DEBIAN_FRONTEND=noninteractive
command -v curl >/dev/null && command -v cc >/dev/null && command -v pkg-config >/dev/null ||
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates build-essential pkg-config git >/dev/null; }
export RUSTUP_HOME=/cache/rustup CARGO_HOME=/cache/cargo PATH=/cache/cargo/bin:$PATH
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --no-modify-path --profile minimal
cd /src && cargo $CARGO_ARGS
chown -R "$HOST_UID:$HOST_GID" /target /cache /src/Cargo.lock'

for v in "${variants[@]}"; do
  args=$(cargo_args "$v")
  suffix=""; [ "$v" = cuda ] && suffix="-cuda12"
  asset="tang-llm-$triple$suffix.tar.gz"
  tdir="${CARGO_TARGET_DIR:-$HOME/build/target-release-$v}"
  mkdir -p "$tdir"
  echo "release-linux: $asset (target dir $tdir)"
  if [ "$native" = 1 ]; then
    # shellcheck disable=SC2086
    (cd "$root" && CARGO_TARGET_DIR="$tdir" cargo $args)
  else
    image=$CPU_IMAGE; [ "$v" = cuda ] && image=$CUDA_IMAGE
    cache="${RELEASE_CACHE:-$HOME/build/release-cache}"; mkdir -p "$cache"
    docker run --rm -v "$root:/src" -v "$tdir:/target" -v "$cache:/cache" \
      -e CARGO_TARGET_DIR=/target -e CARGO_ARGS="$args" \
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$image" bash -c "$in_container"
  fi
  stage=$(mktemp -d)
  cp "$tdir/$triple/release/tang-llm" "$stage/"
  strip "$stage/tang-llm" 2>/dev/null || true
  (cd "$out" && tar -C "$stage" -czf "$asset" tang-llm && sha256sum "$asset" > "$asset.sha256")
  rm -rf "$stage"
  echo "release-linux: $out/$asset ($(du -h "$out/$asset" | cut -f1)), glibc floor $(
    objdump -T "$tdir/$triple/release/tang-llm" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
  if [ -n "$tag" ]; then
    gh release view "$tag" >/dev/null 2>&1 || gh release create "$tag" --title "$tag" --notes "tang $tag"
    gh release upload "$tag" "$out/$asset" "$out/$asset.sha256" --clobber
  fi
done
