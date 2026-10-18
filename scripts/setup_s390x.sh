#!/usr/bin/env bash
# One-command environment setup for the LinuxONE Community Cloud VM (Ubuntu, s390x).
# Status: Planned until run on the VM; see docs/evidence/install_log.md for real runs.
#
#   bash scripts/setup_s390x.sh            full setup (apt + venv + pip + timing test)
#   bash scripts/setup_s390x.sh --check    only record the environment and run the timing test
#
# Time-box: each failing library gets at most LIB_TIMEOUT seconds (default 900 = 15 min),
# then we fall back to the apt package and write down what happened.
# Idempotent: safe to run twice. Never touches secrets. Needs sudo for apt only.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$ROOT/.venv}"
LOG="$ROOT/docs/evidence/install_log.md"
LIB_TIMEOUT="${LIB_TIMEOUT:-900}"
MODE="${1:-full}"
mkdir -p docs

say()  { printf '%s\n' "$*"; }
note() { printf '%s\n' "$*" >> "$LOG"; }
stamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ ! -f "$LOG" ]; then
  note "# Install log"
  note ""
  note "Appended by scripts/setup_s390x.sh. Every entry is a real run on the machine named in it."
  note ""
fi

note "## Run $(stamp)"
note "- mode: ${MODE}"
note "- host: $(uname -srm)"
note "- os: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
note "- nproc: $(nproc)"
note "- memory: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
note "- python3: $(python3 --version 2>&1)"
note "- byte order: $(python3 -c 'import sys; print(sys.byteorder)' 2>&1)"
case "$(uname -m)" in
  s390x) note "- platform check: s390x (OK)";;
  *)     note "- platform check: NOT s390x (this run says nothing about the VM)";;
esac

step() {  # step "name" command...  -> logs OK/FAILED and elapsed seconds
  local name="$1"; shift
  local t0=$SECONDS
  say ">> $name"
  if "$@"; then note "- $name: OK ($((SECONDS - t0)) s)"; return 0
  else          note "- $name: FAILED after $((SECONDS - t0)) s"; return 1; fi
}

if [ "$MODE" != "--check" ]; then
  # 1. system packages (build tools only matter if a wheel has to be built from source)
  APT_PKGS="python3-venv python3-dev python3-pip build-essential gfortran pkg-config \
libopenblas-dev liblapack-dev cmake git curl"
  if command -v apt-get >/dev/null 2>&1; then
    step "apt update"  sudo apt-get update -y
    step "apt install base packages" sudo apt-get install -y $APT_PKGS
  else
    note "- apt: not available, skipped"
  fi

  # 2. venv. --system-site-packages lets the apt fallback below be seen from the venv.
  if [ ! -d "$VENV" ]; then
    step "create venv" python3 -m venv --system-site-packages "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  step "upgrade pip" python -m pip install --upgrade pip

  # 3. each requirement, wheels only first, then apt fallback, then (last) source build
  pkg_import() { case "$1" in scikit-learn) echo sklearn;; *) echo "$1";; esac; }
  pkg_apt()    { case "$1" in
                   scikit-learn) echo python3-sklearn;; matplotlib) echo python3-matplotlib;;
                   streamlit) echo "";;  *) echo "python3-$1";; esac; }
  while read -r line; do
    pkg="${line%%#*}"; pkg="$(echo "$pkg" | xargs)"
    [ -z "$pkg" ] && continue
    mod="$(pkg_import "$pkg")"
    if python -c "import $mod" 2>/dev/null; then note "- $pkg: already importable"; continue; fi
    if step "pip wheel: $pkg" timeout "$LIB_TIMEOUT" python -m pip install --only-binary=:all: "$pkg"; then continue; fi
    apt_name="$(pkg_apt "$pkg")"
    if [ -n "$apt_name" ] && command -v apt-get >/dev/null 2>&1; then
      note "  workaround: no s390x wheel for $pkg, trying apt package $apt_name"
      if step "apt fallback: $apt_name" sudo apt-get install -y "$apt_name"; then continue; fi
    fi
    note "  last resort: building $pkg from source (slow, time-boxed ${LIB_TIMEOUT}s)"
    step "pip source build: $pkg" timeout "$LIB_TIMEOUT" python -m pip install "$pkg" \
      || note "  UNRESOLVED: $pkg. Stop here and record it; do not keep retrying."
  done < "$ROOT/requirements.txt"
else
  [ -d "$VENV" ] && source "$VENV/bin/activate"
fi

# 4. versions actually installed
VERS="$(python - <<'PY' 2>&1
import importlib
out = []
for m in ["numpy", "pandas", "sklearn", "scipy", "joblib", "matplotlib", "streamlit", "pytest"]:
    try:
        out.append(f"{m} {importlib.import_module(m).__version__}")
    except Exception as e:
        out.append(f"{m} MISSING")
print(", ".join(out))
PY
)"
note "- versions: $VERS"

# 5. timing test: fit one forest with the PRD parameters on random data and time it
step "timing test (import + one RandomForest fit)" python - <<'PY'
import time, sys
t0 = time.perf_counter()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
t1 = time.perf_counter()
rng = np.random.default_rng(42)
X = rng.normal(size=(2500, 13)); y = (X[:, 0] + rng.normal(size=2500) > 0).astype(int)
t2 = time.perf_counter()
RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42).fit(X, y)
t3 = time.perf_counter()
print(f"import {t1-t0:.2f}s, fit(2500 rows, 100 trees) {t3-t2:.2f}s, byteorder {sys.byteorder}")
PY
TIMING_EXIT=$?

note ""
say "Environment recorded in $LOG"
exit $TIMING_EXIT
