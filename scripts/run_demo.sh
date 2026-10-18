#!/usr/bin/env bash
# One-command demo (FR-11).
#   bash scripts/run_demo.sh            full demo, exits 0 and prints the receipt path
#   bash scripts/run_demo.sh --no-guard skip the guard stream
# Steps: generate, train, baseline audit, guard drift stream, erase of a member-canary batch,
# after-audit, report, verification. Same seed (42) gives the same attack numbers; compare
# receipts/demo_attack_numbers.json between runs.
# Needs the Python deps from requirements.txt (run scripts/setup_s390x.sh first).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -f .venv/bin/activate ] && source .venv/bin/activate

# Throwaway HMAC key for THIS run only. Never written to disk, never committed.
FORGETME_HMAC_KEY="demo-$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export FORGETME_HMAC_KEY

echo "ForgetMe + Aegis-Z Guard demo | $(uname -m) | $(nproc) core(s) | $(python3 --version 2>&1)"
python3 scripts/run_demo.py "$@"
