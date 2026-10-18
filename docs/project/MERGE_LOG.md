# MERGE_LOG

The original per-module merge notes described a copy-assembly of separate drops and are kept only in git history.
This file records what changed in the final fix round, by feature. Decisions are in `SPEC_CHANGES.md`.

| Feature | Change |
|---|---|
| Repository | One folder with `src/`, `app/`, `scripts/`, `bench/`, `tests/`, `docs/`; pytest markers `slow` and `linux_only`; generated runtime folders are git-ignored; stale copies archived outside the repository |
| Guard on the receipt | `guard.drift_flags` and `guard.held_decisions` are counts of distinct village flags and distinct held decisions, de-duplicated across the overlapping window; the facade, queue, demo and UI pass them through; ids stay in memory |
| After-erase check (SC-13) | Subset-level assertions are one-sided with a power check; the shard-level calibration asymmetry is measured and disclosed |
| Audit gate | Floors are checked against the calibrated AUC; raw and calibrated are printed side by side |
| Benchmarks | `audit_scaling` is timed with a cold shadow cache; warm-cache rows are labelled separately; each machine has its own snapshot |
| Resources test | Caps with `min(2, cores)` so it holds on one or two cores |
| Optional components | ONNX and differential-privacy attempts in dedicated venvs, logged |
| Readiness | VM runbook script, s390x risks, Python 3.10 syntax check |
| Honesty tooling | `scripts/check_claims.py`, `scripts/check_hygiene.py`, `scripts/final_check` |
