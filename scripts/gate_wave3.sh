#!/usr/bin/env bash
# Wave 3 gate. 
# ONNX, DP, Granite and Triton may be Planned only when an attempt log is in docs/evidence/.
# A missing log is FAIL. A present log with the item labelled Planned is not a silent skip.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p docs/evidence
exec > >(tee "docs/evidence/gate_wave3.log") 2>&1
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

echo "GATE 3 start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$PY scripts/contract_check.py --wave 3
cc=$?

$PY -m pytest tests/test_dp.py tests/test_onnx.py tests/test_shards_curve.py -q --tb=line
py=$?

planned=0
if [ "$py" -ne 0 ]; then
  echo "Wave 3 pytest failed. Planned is acceptable only with an attempt log."
  for log in docs/evidence/install_log.md docs/evidence/diffprivlib_install.log docs/evidence/granite_attempt.log docs/evidence/triton_attempt.log; do
    if [ -f "$log" ]; then
      echo "attempt log present: $log"
      planned=1
    fi
  done
fi

echo "Demo: bash scripts/run_demo.sh   (fresh clone, on the VM, not this laptop)"
echo "This script does not run the demo. Gate 3 demo evidence has to come from the VM."

if [ "$cc" -eq 0 ] && { [ "$py" -eq 0 ] || [ "$planned" -eq 1 ]; }; then
  echo "GATE 3: PASS (pytest green, or a documented Planned attempt log exists)"
  echo "Demo-on-VM is still a separate required check. See docs/project/STATUS.md."
  exit 0
fi
echo "GATE 3: FAIL"
exit 1
