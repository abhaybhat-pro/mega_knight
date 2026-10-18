# IBM / platform status: Done, Measured, Planned

Rule: **Done** = built and run, evidence named. **Measured** = a number we recorded on the
target machine. **Planned** = not run yet. Nothing here is claimed until it has run.
Telum, Spyre and Secure Execution are **not claimed** anywhere in this project.

Numbers below come from named x86_64 rehearsal machines (WSL2, 8 cores, 3807 MB RAM; and a 2-core cloud container, tabulated separately). None is an s390x measurement. Benchmark evidence: `docs/evidence/bench_snapshot_*`.

| Item | Status | Evidence / what is missing |
|---|---|---|
| Code runs on the LinuxONE s390x VM | **Planned** | Never run on s390x yet. First action: `bash scripts/setup_s390x.sh`, then the entry it appends to `install_log.md` is the evidence. |
| `scripts/setup_s390x.sh` | **Done** (software) / **Planned** (on VM) | Syntax-checked; `--check` mode run once on x86_64 (see `install_log.md`, which says "NOT s390x"). The apt/venv/pip fallbacks are untested on s390x. |
| Big-endian correctness of the model hash | **Planned** | The shard layer hashes little-endian float64 bytes on purpose; this has only run on little-endian x86_64. Check `pytest tests/test_shards.py` on the VM. |
| Benchmark harness `bench/run_benchmarks.py` | **Done** (software) | `audit_scaling` calls real `run_audit` (`results_meta.json`, stand-in null). |
| Benchmark numbers (`bench/results.csv`) | **Measured** on WSL x86_64, **Planned** on s390x | 18 Oct 2026, 8 cores, median of 3, `training_file=train.csv`. Train 8 shards: 4.767 s at 1 worker, 1.721 s at 8. Parallel full retrain 1.921 s, serial 2.929 s, one shard 0.335 s. Speedup vs parallel baseline 5.73x. Speedup vs serial 8.73x (labelled: one shard versus every shard one at a time). |
| Resource profile | **Measured** on WSL x86_64, **Planned** on s390x | Cold shadow build 23.28 s, avg CPU 95.5%, peak 100%, peak RSS 278 MB. `bench/resource_profile.csv`. |
| Hour-3 audit gate | **Measured** on WSL x86_64, **Planned** on s390x | `docs/evidence/audit_gate_local.log`. Shard worst calibrated AUC 0.988, ensemble calibrated 0.825 (interval 0.797–0.854), raw 0.613, TPR@1%FPR 0.165. GATE PASS. Evidence, not a proof. |
| Numbers validator + template report | **Done** | `tests/test_validator.py`: passes on the template report, fails and names an injected number. |
| IBM Granite via llama.cpp (`granite_report`) | **Planned** | Plumbing tested only with a fake llama.cpp script. No llama.cpp build, no Granite model has run. Verify a big-endian Granite GGUF actually exists before assuming it (see `evidence/granite_attempt.log`). |
| ONNX export of the 8 shards | **Done** on x86_64 rehearsal, **Planned** on s390x | `docs/evidence/onnx_env.md`. Parity with scikit-learn: max difference 2.9e-07 (tolerance 1e-5). Dedicated venv, not the product venv. |
| Differential-privacy comparison (diffprivlib) | **Done** on x86_64 rehearsal, **Planned** on s390x | `docs/evidence/dp_env.md`. A separate guarantee from unlearning. Needs scikit-learn 1.5.2 in its own venv. |
| IBM Z Deep Learning Compiler (ZDLC) on the ONNX models | **Planned** | Not attempted. `triton_plan.md` lists the intended steps; they are unverified. |
| Triton Inference Server serving | **Planned** | Not attempted. Depends on what the VM allows (containers, registries). |
| One-command demo `scripts/run_demo.sh` on a fresh clone on the VM | **Measured** on WSL x86_64, **Planned** on s390x | Two runs, both exit 0, `demo_attack_numbers.json` identical. Shard attack AUC 0.988 to 0.457 (drop 0.532). Ensemble on those 41 members 0.774 to 0.513. Writer: template. Validator PASS. Signature and log chain OK inside the run. |
| Telum on-chip AI accelerator | **Not claimed** | Not used, not benchmarked. |
| Spyre accelerator | **Not claimed** | Not used. |
| Secure Execution | **Not claimed** | Not used. |

Update this table when something actually runs, and paste the evidence path next to it.

**Known limitation (SC-13).** Post-erase shard-level attack AUC lands at or below chance because of a calibration asymmetry (held-out canaries are calibrated with the statistics of the shard they hash to, members with their own shard's). 9 seeds, x86_64 rehearsal: ensemble mean 0.494, shard-level calibrated mean 0.478, raw mean 0.499. Matched held-out-only pseudo-members reproduce the shift, so it is a calibration effect, not leftover membership signal. Evidence: `docs/evidence/sc13_diagnostics/`.
