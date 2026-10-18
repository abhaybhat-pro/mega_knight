# ForgetMe + Aegis-Z Guard. Run from the repository root. Needs a POSIX shell.
PY      ?= python3
VENV    ?= .venv
BIN      = $(VENV)/bin
PYTEST   = $(BIN)/python -m pytest

.PHONY: setup test test-all gate demo ui bench clean

setup:                      ## create .venv and install the core requirements
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt

test:                       ## fast tests only (a few minutes)
	$(PYTEST) -m "not slow" -q

test-all:                   ## the whole suite, including slow erase and audit tests (run detached)
	$(PYTEST) -q

gate:                       ## contract + structure check, audit gate, claims check
	$(BIN)/python scripts/contract_check.py --wave 3
	$(BIN)/python scripts/audit_gate.py
	$(BIN)/python scripts/check_claims.py

demo:                       ## the whole story with a throwaway HMAC key (see scripts/run_demo.sh)
	bash scripts/run_demo.sh

ui:                         ## Streamlit UI (needs FORGETME_HMAC_KEY in the environment to sign)
	$(BIN)/python -m streamlit run app/streamlit_app.py

bench:                      ## benchmarks and resource profile (writes git-ignored files under bench/)
	$(BIN)/python bench/run_benchmarks.py --profile

clean:                      ## remove caches and generated runtime files, keep .gitkeep
	find . -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find data models logs receipts -mindepth 1 ! -name .gitkeep -delete
