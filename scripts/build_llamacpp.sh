#!/usr/bin/env bash
# Build llama.cpp (CPU only) on the s390x VM.  Status: Planned until run.
# Needs network ONLY for this build step (git clone) and model download, never for the app.
# Usage: bash scripts/build_llamacpp.sh [target_dir]   (default: $HOME/llama.cpp)
set -euo pipefail
DEST="${1:-$HOME/llama.cpp}"
LOG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/evidence/granite_attempt.log"
mkdir -p "$(dirname "$LOG")"
log() { printf '%s | BUILD | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
log "host $(uname -srm), nproc $(nproc)"
[ -d "$DEST/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$DEST" || { log "git clone FAILED"; exit 1; }
cd "$DEST"
log "commit $(git rev-parse --short HEAD)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON >/dev/null || { log "cmake configure FAILED"; exit 1; }
cmake --build build --config Release -j "$(nproc)" || { log "cmake build FAILED"; exit 1; }
BIN="$(ls build/bin/llama-cli 2>/dev/null || true)"
[ -n "$BIN" ] || { log "build finished but build/bin/llama-cli not found"; exit 1; }
log "built OK: $DEST/$BIN"
echo "export FORGETME_LLAMA_CLI=$DEST/$BIN"
echo "export FORGETME_GRANITE_GGUF=/path/to/big-endian-granite.gguf   # must be a BIG-ENDIAN GGUF"
