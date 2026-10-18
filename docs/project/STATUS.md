# STATUS

State as of the final verification run on an x86_64, 2-core cloud container (Python 3.13 for the main suite, Python 3.12 for the
minimum-version run). This is not an s390x result. The LinuxONE host stays `<TODO>` until a person supplies the SSH line.
Evidence for every line below is under `docs/evidence/` (see `docs/evidence/final_verification/SUMMARY.md`).

Labels: Done / Measured / Planned. Measured numbers name their machine.

## Features

| Feature | State | Evidence |
|---|---|---|
| Deterministic synthetic data (seed 42) | Done | `tests/test_generate.py`; two generations give identical CSV hashes (V5) |
| Eight RandomForest shards, one-shard retrain on erase | Done | `tests/test_shards.py`, `tests/test_erase.py`; 1 vs 4 workers give identical hashes |
| Membership-inference audit, before and after | Measured (x86_64) | `tests/test_audit.py` 7 passed; gate: worst-shard calibrated AUC 0.988, whole-model calibrated 0.825 vs raw 0.613 (`final_verification/V4.log`) |
| After-erase assertions (SC-13) | Done | Subset-level checks are one-sided with a power check; disclosed in README limitations |
| Signed receipt and hash-chained log | Measured (x86_64) | `tests/test_receipt.py`, `tests/test_log.py`; byte-flip tamper tests detected (V6) |
| Aegis-Z Guard (drift, entropy, subgroup AUROC; hold for review) | Done | `tests/test_guard.py`; holds = persons in villages with max PSI >= 0.25 (SC-9); counts-only on the receipt |
| Coalescing erase queue | Done | `tests/test_queue.py` |
| Plain-language report and numbers validator | Done | `tests/test_validator.py`; template writer |
| Streamlit UI | Done | real-data page check with and without plotly, health endpoint ok (V7) |
| Benchmarks and resource profile | Measured (named machines) | `docs/evidence/bench_snapshot_*`; audit scaling is timed cold |
| Accuracy cost of sharding | Measured (x86_64) | `docs/evidence/shards_curve/results.csv` |
| ONNX export of the shards | Done on x86_64 | `docs/evidence/onnx_env.md` |
| Differential-privacy comparison | Done on x86_64 | `docs/evidence/dp_env.md`; a separate guarantee from unlearning |
| s390x readiness (runbook, risks, Python 3.10 syntax check) | Done (software) | `scripts/vm_runbook.sh`, `docs/vm_runbook.md`, `docs/s390x_risks.md` |
| Run on the LinuxONE s390x VM | Planned | Never run on s390x |
| IBM Granite via llama.cpp, IBM Z Deep Learning Compiler, Triton | Planned | `docs/ibm_status.md` |
| Telum, Spyre, Secure Execution | Not claimed | Not used |

## Final verification (V1 to V10)

See `docs/evidence/final_verification/SUMMARY.md`. V2 ran the full suite on the latest packages; V3 ran generate, shards, predict,
receipt, log, validator, guard, facade, erase and queue on the pinned minimum versions.
