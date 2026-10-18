#!/usr/bin/env bash
# Export a checkpoint zip and git bundle of HEAD (tracked files only) into the given folder.
set -euo pipefail
out="${1:-/tmp/claude-0/out}"; mkdir -p "$out"
git archive --format=zip --prefix=forgetme-aegis/ -o "$out/forgetme-aegis-checkpoint.zip" HEAD
git bundle create "$out/forgetme-aegis-checkpoint.bundle" --all >/dev/null 2>&1
sha256sum "$out"/forgetme-aegis-checkpoint.*
