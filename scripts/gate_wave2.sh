#!/usr/bin/env bash
# Wave 2 gate. Requires Gate 1 already green.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p docs/evidence
exec > >(tee "docs/evidence/gate_wave2.log") 2>&1
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

echo "GATE 2 start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$PY scripts/contract_check.py --wave 2
cc=$?

$PY -m pytest \
  tests/test_erase.py \
  tests/test_queue.py \
  tests/test_guard.py \
  tests/test_subgroup_after_erase.py \
  tests/test_validator.py \
  tests/test_integration.py \
  -q --tb=line
py_rest=$?

# -k must not filter the files above. It selects the two audit acceptance tests only.
$PY -m pytest tests/test_audit.py -k "after_erase or single_shard" -q --tb=line
py_audit=$?
py=0
if [ "$py_rest" -ne 0 ] || [ "$py_audit" -ne 0 ]; then
  py=1
fi

if [ -f scripts/smoke_ui.py ]; then
  $PY scripts/smoke_ui.py
  ui=$?
else
  echo "MISSING scripts/smoke_ui.py"
  ui=1
fi

if [ "$cc" -eq 0 ] && [ "$py" -eq 0 ] && [ "$ui" -eq 0 ]; then
  echo "GATE 2: PASS"
  exit 0
fi
echo "GATE 2: FAIL"
exit 1
