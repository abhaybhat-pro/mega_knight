#!/usr/bin/env bash
# Wave 1 gate. Prints PASS or FAIL. Does not soften thresholds.
# Hour-3 audit_gate.py is reported separately: it is a VM measurement, not a unit test.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p docs/evidence
exec > >(tee "docs/evidence/gate_wave1.log") 2>&1
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

echo "GATE 1 start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "python: $($PY --version 2>&1)"

$PY scripts/contract_check.py --wave 1
cc=$?

$PY -m pytest \
  tests/test_generate.py \
  tests/test_shards.py \
  tests/test_predict.py \
  tests/test_receipt.py \
  tests/test_log.py \
  tests/test_audit.py -k "not after_erase and not single_shard" \
  -q --tb=line
py=$?

echo "HOUR-3 audit_gate.py is not part of this script's pytest list."
echo "On the LinuxONE VM, after generate + train, run: $PY scripts/audit_gate.py"
echo "Do not treat a laptop result as the s390x measurement."

if [ "$cc" -eq 0 ] && [ "$py" -eq 0 ]; then
  echo "GATE 1: PASS"
  exit 0
fi
echo "GATE 1: FAIL"
exit 1
