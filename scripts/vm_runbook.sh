#!/usr/bin/env bash
# One command to run on the LinuxONE s390x VM. Records evidence under docs/evidence/vm_run_<utc>/.
# Status: Planned until a person runs it on the VM. Nothing here was run on s390x by the author.
#
#   bash scripts/vm_runbook.sh            everything
#   bash scripts/vm_runbook.sh --skip-setup   skip scripts/setup_s390x.sh (venv already built)
#
# Each step logs its own file and exit code; a failed step does not stop the run, so one
# pass shows every problem. The final SUMMARY.md says what ran, what failed, and which host
# produced it. A run on a host that is not s390x writes "NOT s390x" at the top of everything.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/docs/evidence/vm_run_$TS"; mkdir -p "$OUT"
ARCH="$(uname -m)"; NPROC="$(nproc)"
SUMMARY="$OUT/SUMMARY.md"
{ echo "# VM run $TS"; echo
  echo "- host: $(uname -srm)  cores: $NPROC  python: $(python3 --version 2>&1)  byteorder: $(python3 -c 'import sys;print(sys.byteorder)')"
  if [ "$ARCH" = s390x ]; then echo "- platform: s390x"; else echo "- platform: **NOT s390x** ($ARCH). Rehearsal only, not a VM result."; fi
  echo; echo "| step | exit | log |"; echo "|---|---|---|"; } > "$SUMMARY"

run() {  # run name cmd...
  local name="$1"; shift
  echo ">> $name"
  "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  echo "| $name | $rc | $name.log |" >> "$SUMMARY"
  return 0
}

[ "${1:-}" = "--skip-setup" ] || run setup bash scripts/setup_s390x.sh
[ -f .venv/bin/activate ] && source .venv/bin/activate
# Throwaway key for this run only; never written to disk.
FORGETME_HMAC_KEY="vm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')"; export FORGETME_HMAC_KEY

run versions      python -c "import numpy,pandas,sklearn,scipy,joblib,sys;print(sys.version);print(numpy.__version__,pandas.__version__,sklearn.__version__,scipy.__version__,joblib.__version__,sys.byteorder)"
run compile_all   python -m compileall -q src app scripts bench tests
run py310_syntax  python scripts/check_py310_syntax.py
run tests_fast    python -m pytest -m "not slow" -q -p no:cacheprovider
run tests_endian  python -m pytest tests/test_shards.py -q -p no:cacheprovider
run contract      python scripts/contract_check.py --wave 3
run audit_gate    python scripts/audit_gate.py
run demo          bash scripts/run_demo.sh
run shards_curve  python -m src.model.shards_curve
SNAP="docs/evidence/bench_snapshot_${ARCH}-${NPROC}cores"
run bench         python bench/run_benchmarks.py --out "$SNAP" --profile
run tests_slow    python -m pytest -m slow -q -p no:cacheprovider
run claims        python scripts/check_claims.py
cp -r bench/shards_curve "$SNAP/" 2>/dev/null || true
echo; echo "Evidence in $OUT. Read SUMMARY.md. Commit docs/evidence/ if the run is real."
cat "$SUMMARY"
