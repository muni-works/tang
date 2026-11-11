#!/usr/bin/env bash
# Mirror of /models/bootstrap/tang-up.sh on the frog-models global volume; keep the two in sync.
# tang-up: take a fresh RunPod pod (runpod/pytorch:*-cu1281-*-ubuntu2404) from boot
# to building tang in about a minute, using the global volume mounted at /models.
#
#   source /models/bootstrap/tang-up.sh     # sets PATH/env in the current shell
#   bash   /models/bootstrap/tang-up.sh     # sets up; then `source ~/.tang-up/env`
#
# Idempotent: re-running only refreshes what is missing or stale.
#
# What lives on the volume (never credentials):
#   /models/bootstrap/rust-toolchain.tar.zst   stable rustc, cargo, rustfmt, sccache
#   /models/bootstrap/sccache/                 shared sccache local-disk cache
#   /models/bootstrap/sccache.tar.zst          (TANG_SCCACHE=sync only)
#
# Knobs (env):
#   TANG_UP_HOME   local toolchain dir           (default ~/.tang-up)
#   TANG_DIR       tang checkout                 (default ~/tang)
#   TANG_REF       branch to check out           (default main)
#   TANG_SCCACHE   volume | sync | off           (default volume)
#     volume: sccache's disk backend directly on /models (content-addressed, so a
#             stale or missing entry from another pod is just a miss)
#     sync:   per-pod cache on local disk, pulled from the volume tarball here;
#             push it back at the end with `tang-up-push`
#   SCCACHE_CACHE_SIZE (default 40G)
#
# Rebuild the toolchain tarball (on any pod with this image):
#   S=$(mktemp -d); RUSTUP_HOME=$S/rustup CARGO_HOME=$S/cargo sh <(curl -sSf https://sh.rustup.rs) \
#     -y --no-modify-path --profile minimal --default-toolchain stable -c rustfmt
#   # + sccache musl binary into $S/cargo/bin, then:
#   tar -C $S -cf - rustup cargo | zstd -T8 -3 -o /models/bootstrap/rust-toolchain.tar.zst

_tang_up() {
  local t0=$SECONDS
  local vol=/models/bootstrap
  local home=${TANG_UP_HOME:-$HOME/.tang-up}
  local dir=${TANG_DIR:-$HOME/tang}
  local ref=${TANG_REF:-main}
  local mode=${TANG_SCCACHE:-volume}
  local tarball=$vol/rust-toolchain.tar.zst

  [ -f "$tarball" ] || { echo "tang-up: $tarball missing" >&2; return 1; }

  # 1. toolchain: unpack once; re-unpack when the volume tarball changes
  local stamp
  stamp=$(stat -c '%s-%Y' "$tarball")
  if [ "$(cat "$home/.stamp" 2>/dev/null)" != "$stamp" ]; then
    rm -rf "$home.new" && mkdir -p "$home.new"
    zstd -dc "$tarball" | tar -xf - -C "$home.new" || return 1
    echo "$stamp" > "$home.new/.stamp"
    rm -rf "$home" && mv "$home.new" "$home"
    echo "tang-up: toolchain unpacked ($((SECONDS - t0))s)"
  fi

  # 2. cache
  local cache
  case $mode in
    volume) cache=$vol/sccache ;;
    sync)
      cache=$home/sccache
      if [ ! -d "$cache" ] && [ -f "$vol/sccache.tar.zst" ]; then
        mkdir -p "$cache" && zstd -dc "$vol/sccache.tar.zst" | tar -xf - -C "$cache"
        echo "tang-up: sccache pulled ($(du -sh "$cache" | cut -f1))"
      fi ;;
    off) cache= ;;
    *) echo "tang-up: TANG_SCCACHE=$mode? (volume|sync|off)" >&2; return 1 ;;
  esac

  # 3. env file, sourced here and by later shells
  {
    echo "export RUSTUP_HOME=$home/rustup CARGO_HOME=$home/cargo"
    echo "export PATH=$home/cargo/bin:\$PATH"
    if [ -n "$cache" ]; then
      mkdir -p "$cache"
      echo "export RUSTC_WRAPPER=sccache SCCACHE_DIR=$cache SCCACHE_CACHE_SIZE=${SCCACHE_CACHE_SIZE:-40G}"
    else
      echo "unset RUSTC_WRAPPER"
    fi
    # push the per-pod cache back (sync mode); write tmp then rename
    echo "tang-up-push() { [ \"\${TANG_SCCACHE:-$mode}\" = sync ] || { echo 'tang-up-push: sync mode only'; return 0; }; sccache --stop-server >/dev/null 2>&1; tar -C $home/sccache -cf - . | zstd -T8 -3 -q -o $vol/sccache.tar.zst.\$\$ && mv -f $vol/sccache.tar.zst.\$\$ $vol/sccache.tar.zst; }"
  } > "$home/env"
  # shellcheck disable=SC1091
  . "$home/env"
  grep -q tang-up/env "$HOME/.bashrc" 2>/dev/null || echo "[ -f $home/env ] && . $home/env" >> "$HOME/.bashrc"

  # 4. tang over HTTPS (public repo; no credentials)
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch -q --no-prune origin "$ref" && git -C "$dir" checkout -q "$ref" \
      && git -C "$dir" merge -q --ff-only "origin/$ref" \
      || echo "tang-up: $dir not fast-forwarded (local changes?); left as is" >&2
  else
    git clone -q --branch "$ref" https://github.com/muni-works/tang "$dir" || return 1
  fi

  echo "tang-up: $(rustc -V | cut -d' ' -f1-2), sccache=${cache:-off}, $dir @ $(git -C "$dir" rev-parse --short HEAD)"
  echo "tang-up: ready in $((SECONDS - t0))s"
}

_tang_up
# shellcheck disable=SC2181
_tang_up_rc=$?
unset -f _tang_up
if (return 0 2>/dev/null); then return $_tang_up_rc; else exit $_tang_up_rc; fi
