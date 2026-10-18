# ForgetMe + Aegis-Z Guard

IBM Z Datathon 2026, Theme 2 (AI Secured, Innovation Without Exposure).

A small rural clinic honours a patient's "forget me" request by retraining **one** of 8
shards of a RandomForest triage model. ForgetMe backs the erasure with a membership-inference
audit (400 member and 400 held-out canaries, 16 shadow models) and an HMAC-signed receipt in a
hash-chained log. Aegis-Z Guard watches for drift and subgroup (village, age band) confidence
drops, holds risky decisions for human review, and checks that an erasure did not hurt any
subgroup. Everything is built for a LinuxONE (s390x, big-endian) Ubuntu VM. **Synthetic data
only.**

> The audit is empirical evidence from simulated attacks, not a proof of forgetting.
> Nothing here is legal advice. The legal status of model weights under the DPDP Act is
> unsettled, and DPDP dates are to be verified.

## 0. This laptop (WSL2 Ubuntu, x86_64 rehearsal)

These steps were run here. They are not an s390x result. The LinuxONE host stays `<linux1@148.100.78.177>` until a person supplies the SSH line.

```bash
# from the Windows tree, the Linux copy lives at ~/forgetme-aegis
cd ~/forgetme-aegis
uv venv --python python3.12 .venv          # python3-venv is not installed; sudo needs a password
source .venv/bin/activate
uv pip install numpy pandas scikit-learn scipy joblib streamlit matplotlib plotly pytest
export FORGETME_HMAC_KEY=...               # throwaway for a demo; never commit the value
python -m src.data.generate
python scripts/audit_gate.py
python bench/run_benchmarks.py --profile
bash scripts/run_demo.sh
```

Native Windows can run the unit tests that do not need `fork`, `/proc`, or `nproc`. The rehearsal numbers below were taken in WSL, not in native Windows. `docs/local_env.md` records Python 3.12.3, `x86_64`, 8 cores, 3807 MB RAM, and the package freeze.

## 1. Get on the VM and set up (Planned until run on s390x)

Use the SSH details IBM gave you for your LinuxONE Community Cloud instance
(`ssh <user>@<vm-address>`; fill in the real values for the team here: `<TODO>`).

    git clone <repo-url> forgetme-aegis && cd forgetme-aegis
    bash scripts/setup_s390x.sh          # apt, venv, pip, timing test; logs to docs/evidence/install_log.md
    source .venv/bin/activate

What the script does on s390x, in order, each step time-boxed (`LIB_TIMEOUT`, default 15 min):

1. `apt install` build tools and BLAS/LAPACK (`python3-venv`, `build-essential`, `gfortran`,
   `libopenblas-dev`, ...).
2. Create `.venv` with `--system-site-packages`, so apt-installed packages are visible.
3. For each line of `requirements.txt`: pip **wheel only** first; if none exists for s390x,
   the apt package (`python3-sklearn`, `python3-scipy`, ...); last resort a source build.
4. Record versions, `nproc`, byte order and a one-forest timing test.

Known workarounds are whatever `docs/evidence/install_log.md` says happened on the VM; that file is
written by real runs, not by hand. `streamlit` has no apt package: if its wheel will not
install, the UI is the only thing affected. Run `bash scripts/setup_s390x.sh --check` to
re-record the environment without installing.

## 2. Run

    python -m src.data.generate           # data/*.csv (git-ignored)
    bash scripts/run_demo.sh              # the whole story, exits 0, prints the receipt path
    pytest                                # all tests

`run_demo.sh` sets a throwaway `FORGETME_HMAC_KEY` for that run only (the real key is never
in the repo), then: generate, train 8 shards, baseline audit, guard drift stream, erase every
member canary in one shard as one request, after-audit, plain-language report, verification
of the receipt signature and the log chain. Same seed (42) gives the same attack numbers:
diff `receipts/demo_attack_numbers.json` between two runs. An existing erasure log is moved
aside (`logs/*.bak`), never deleted.

Optional Granite report (Planned): `bash scripts/build_llamacpp.sh`, then export
`FORGETME_LLAMA_CLI` and `FORGETME_GRANITE_GGUF` (a **big-endian** GGUF). Without them the
report falls back to the template and its first line says so. Whatever writes the report,
`validate_numbers` must find every number in the receipt or the report fails validation and is not used.

## 3. Benchmarks

    python bench/run_benchmarks.py        # writes bench/results.csv, results_meta.json, scaling_notes.md

Columns: `experiment, n_workers, n_shards, seconds, n_cores`. `seconds` is the median of 3
runs. `n_cores` is `available_cores()`. `--profile` writes
`bench/resource_profile.csv`. The audit stand-in is gone: `audit_scaling` calls `run_audit` and is timed COLD (a fresh empty models dir per run, so the shadow build is inside the timed region; `--audit-repeats`, default 1). A warm-cache audit is only scoring and is written as `audit_warm_cache`, never as scaling.
Full retrain is timed twice. The parallel row is the headline clinic baseline. The serial
row is labelled separately. Training file on this run: `train.csv` (`results_meta.json`).

**Machine: WSL2 x86_64, 8 cores, rehearsal (the lead's machine).** 18 Oct 2026, 20,000 training rows, median of 3. Not s390x. Evidence: `docs/evidence/bench_snapshot_wsl2_x86_64-8cores_rehearsal/`. Numbers from other machines go in their own snapshot and are never mixed with this table.

| workers | train_scaling seconds (8 shards) | speedup vs 1 worker |
|---|---|---|
| 1 | 4.767 | 1.00x |
| 2 | 4.330 | 1.10x |
| 4 | 2.818 | 1.69x |
| 8 | 1.721 | 2.77x |

One shard retrain at 8 shards: 0.335 s. Parallel full retrain (`n_workers=-1`, which resolved to the 8 cores): 1.921 s. Serial full retrain: 2.929 s.

- Speedup versus the parallel baseline: 1.921 / 0.335 = 5.73x.
- Speedup versus the serial full retrain: 2.929 / 0.335 = 8.73x. That figure is one shard against retraining every shard one at a time. It is labelled here so it is not read as an 8-core scaling result. The parallel full retrain itself is 2.929 / 1.921 = 1.53x the serial full retrain.

Eight shard tasks on these 8 cores are one wave, so the scheduling ceiling is 100%. On a 3-core VM the same 8 tasks run as waves of 3, 3 and 2, and the ceiling is 8/9 (about 89%). That 3-core figure is the scheduling rule, not a measurement from this machine.

In this WSL snapshot the `audit_scaling` rows were taken after the shadow cache existed, so they are scoring only (0.73 s at 1 worker, 0.70 s at 8) and are not a scaling result; the harness now measures them cold. A cold shadow build, measured in its own process, took 23.28 s on 8 workers, average CPU 95.5%, peak CPU 100% (soft gate PASS). The short train_all row in the profile (2.29 s, average CPU 44%, peak 82%) is below the 85% line because the job is only a few seconds long and pool startup is a large part of the wall clock.

| phase | workers | wall s | avg CPU % | peak CPU % | peak RSS MB | min MemAvailable MB |
|---|---|---|---|---|---|---|
| generate | 1 | 0.21 | 13.2 | 13.2 | 791 | 2054 |
| train_all | 8 | 2.29 | 44.4 | 82.2 | 791 | 1937 |
| shadow_build (cold) | 8 | 23.28 | 95.5 | 100 | 278 | 2049 |
| audit (cache warm) | 8 | 0.70 | 12.4 | 12.4 | 791 | 2023 |
| erase_one_shard | 1 | 0.34 | 12.7 | 12.7 | 791 | 2028 |
| full_retrain_parallel | 8 | 1.77 | 52.4 | 81.9 | 792 | 1962 |
| full_retrain_serial | 1 | 2.97 | 12.5 | 12.6 | 792 | 1971 |
| guard_stream | 1 | 3.34 | 12.4 | 12.6 | 792 | 1982 |

Peak RSS in a multi-phase process is the high-water mark since process start (`ru_maxrss`), so later rows repeat an earlier peak. The cold shadow row is its own process. Host disk in use was about 5.1 GB of the 1007 GB volume. Numbers on the LinuxONE VM are still Planned.

### Second machine: x86_64, 2 cores, cloud container rehearsal

Separate snapshot, never mixed with the WSL2 table above: `docs/evidence/bench_snapshot_x86_64-2cores-cloud-rehearsal/`
(`results.csv`, `results_meta.json`, `resource_profile.csv`, `scaling_notes.md`). 20,000 training rows, median of 3 (audit rows: 1 run). Not s390x.
Two cores say nothing general about speedup, so only raw seconds are quoted.

| measurement | seconds |
|---|---|
| train 8 shards, 1 worker | 2.486 |
| train 8 shards, 2 workers | 1.559 |
| one shard retrain (8 shards) | 0.330 |
| full retrain, parallel (headline) | 1.457 |
| full retrain, serial | 2.432 |
| `audit_scaling` COLD, 1 worker | 41.045 |
| `audit_scaling` COLD, 2 workers | 22.008 |

The cold audit includes building the shadow forests; the resource profile in the same folder records that build at 22.49 s on 2 workers (average CPU 99.2%).

### Accuracy cost of sharding

`python -m src.model.shards_curve` (evidence: `docs/evidence/shards_curve/results.csv`, `curve.png`, log `docs/evidence/shards_curve_x86_64-2cores.log`).
Test AUROC of the mean-of-shards ensemble: 1 shard 0.758, 4 shards 0.774, 8 shards 0.776, 16 shards 0.775. Going from 1 to 8 shards changes test AUROC by +0.018 on synthetic data (no accuracy cost observed here).

## 4. Done / Measured / Planned

The full table, with evidence, is in [`docs/ibm_status.md`](docs/ibm_status.md). In short:
the software and its tests are Done; every number or claim that needs the s390x VM, Granite,
the IBM Z Deep Learning Compiler or Triton is Planned until it has run.
Telum, Spyre and Secure Execution are not used and not claimed.

## 5. Limitations

- Synthetic data; real clinical data would need consent, governance and a legal review.
- The membership-inference audit tests specific attacks with a fixed set of canaries. A
  stronger attack could do better. It is evidence, not proof.
- Retraining a shard removes the rows from that shard's model; it does not touch copies of the
  data elsewhere (backups, logs, exports).
- Hash-chained logs detect tampering; they do not prevent deletion of the whole file.
- Plain-language reports are not legal advice.
- Post-erase shard-level attack AUC lands at or below chance (never described as an exact chance value). Cause: a calibration asymmetry (held-out canaries are calibrated with the statistics of the shard they hash to; members with their own shard's). Measured over 9 seeds on an x86_64 rehearsal: ensemble mean 0.494, shard-level calibrated mean 0.478, raw mean 0.499. Matched held-out-only pseudo-members reproduce the shift, so it is a calibration effect, not leftover membership signal. The two subset-level after-erase assertions are therefore one-sided with a power check (SC-13, `docs/project/SPEC_CHANGES.md`).
- Guard holds (SC-9): the hold list is persons in villages whose max feature PSI is at least 0.25. Confidence drops are warnings and age-band drops are advisories; nobody is added to the hold list for them. The original spec also held on a 0.10 confidence drop; the lead chose PSI-only.
- `make_data` returns `(df, {"member": [...], "heldout": [...]})`: the training frame and the person ids of each canary kind.
- Villages 6 and 7 are about 3% of ordinary rows. A 2-batch Guard window (1000 rows) stays under 200 rows for those villages, so they are marked "insufficient data" and are never flagged. `min_subgroup_size` stays 200.
- DPDP dates are to be verified against the gazette. The legal status of model weights under the DPDP Act is unsettled.
- The hour-3 audit gate on this rehearsal passed. Shard worst calibrated AUC 0.988 (mean 0.992), ensemble calibrated AUC 0.825 (95% interval 0.797 to 0.854), ensemble raw AUC 0.613 (interval 0.574 to 0.651), ensemble TPR at 1% FPR 0.165. Local x86 rehearsal, not s390x. The audit is evidence from this attack, not a proof.

## 6. Layout

<!-- TREE:BEGIN -->
```text
forgetme-aegis/
├── app/
│   ├── pages/
│   │   ├── 1_Train.py
│   │   ├── 2_Predict.py
│   │   ├── 3_Erase.py
│   │   ├── 4_Audit.py
│   │   ├── 5_Guard.py
│   │   ├── 6_Receipts.py
│   │   └── 7_Benchmarks.py
│   ├── streamlit_app.py
│   └── ui.py
├── bench/
│   ├── __init__.py
│   └── run_benchmarks.py
├── data/
│   └── .gitkeep
├── docs/
│   ├── evidence/  (logs and benchmark snapshots)
│   ├── project/
│   │   ├── DEMO_RUNBOOK.md
│   │   ├── MERGE_LOG.md
│   │   ├── SPEC_CHANGES.md
│   │   └── STATUS.md
│   ├── submission/
│   │   ├── demo_script.md
│   │   └── facts.md
│   ├── architecture.md
│   ├── demo_recording.md
│   ├── ibm_status.md
│   ├── local_env.md
│   ├── s390x_risks.md
│   ├── triton_plan.md
│   └── vm_runbook.md
├── logs/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── receipts/
│   └── .gitkeep
├── scripts/
│   ├── __init__.py
│   ├── audit_gate.py
│   ├── build_llamacpp.sh
│   ├── check_claims.py
│   ├── check_hygiene.py
│   ├── check_py310_syntax.py
│   ├── contract_check.py
│   ├── demo_pipeline.py
│   ├── diagnose_after_erase.py
│   ├── export_checkpoint.sh
│   ├── final_check
│   ├── gate_wave1.sh
│   ├── gate_wave2.sh
│   ├── gate_wave3.sh
│   ├── gen_tree.py
│   ├── run_demo.py
│   ├── run_demo.sh
│   ├── setup_s390x.sh
│   ├── smoke_ui.py
│   ├── tamper_demo.py
│   ├── verify_demo_integrity.py
│   └── vm_runbook.sh
├── src/
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── mia.py
│   │   └── secondary.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── erase/
│   │   ├── __init__.py
│   │   └── queue.py
│   ├── guard/
│   │   ├── __init__.py
│   │   └── monitor.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dp.py
│   │   ├── shards.py
│   │   └── shards_curve.py
│   ├── report/
│   │   ├── __init__.py
│   │   ├── narrate.py
│   │   └── receipt.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   └── resources.py
│   ├── serving/
│   │   ├── __init__.py
│   │   └── export_onnx.py
│   ├── __init__.py
│   ├── contracts.py
│   └── forgetme_api.py
├── tests/
│   ├── conftest.py
│   ├── fakes_erase.py
│   ├── receipt_fixtures.py
│   ├── test_audit.py
│   ├── test_benchmarks.py
│   ├── test_dp.py
│   ├── test_erase.py
│   ├── test_facade.py
│   ├── test_generate.py
│   ├── test_guard.py
│   ├── test_integration.py
│   ├── test_log.py
│   ├── test_onnx.py
│   ├── test_paths.py
│   ├── test_predict.py
│   ├── test_queue.py
│   ├── test_receipt.py
│   ├── test_resources.py
│   ├── test_shards.py
│   ├── test_shards_curve.py
│   ├── test_subgroup_after_erase.py
│   └── test_validator.py
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── requirements-optional.txt
└── requirements.txt
```
<!-- TREE:END -->

No secrets are committed: the HMAC key comes only from the environment variable
`FORGETME_HMAC_KEY`; model binaries, data CSVs, receipts and logs are git-ignored.
