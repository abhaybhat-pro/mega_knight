# Facts file: the only source for slides, video and form answers

Rule: nothing in the deck, video or form may contain a number or claim that is not in this file.
Every number names its machine and its evidence file. Synthetic data only. Not legal advice.
Numbers for slides come from the LinuxONE s390x VM, which is still Planned: until `scripts/vm_runbook.sh` has run there,
the figures below are x86_64 rehearsal figures and must be labelled that way if shown.

## What the project is
- 8 RandomForest shards; erasing a person retrains only the shard that held them.
- Membership-inference audit (shadow-model attack on 400 member canaries vs 400 held-out canaries) gives before/after evidence. It is empirical evidence from this attack, not a proof.
- Receipts are signed (HMAC-SHA256); the erasure log is a SHA-256 hash chain; both verify in the demo.
- Aegis-Z Guard checks drift, entropy and subgroup AUROC. Outcomes: proceed, warn, hold for human review. Nothing is refused automatically.
- Receipts and logs contain counts and metrics only.

## Demo run: x86_64, 2 cores, cloud container rehearsal (`docs/evidence/demo_run_x86_64-2cores.log`, receipt `demo_receipt_x86_64-2cores.json`)
- Erase request of 41 member canaries in shard 3: one shard of 8 retrained.
- Attack AUC, affected shard: 0.988 before, 0.457 after.
- Attack AUC, whole model (on those 41 members): 0.774 before, 0.513 after (95% intervals 0.692 to 0.852 and 0.421 to 0.609).
- Attacker true-positive rate at 1% false-alarm rate, whole model: 14.6% before, 0.0% after.
- Test AUROC before and after: 0.776 and 0.777. Largest subgroup AUROC drop: 0.002 (limit 0.03).
- Guard drift stream: 1 village reached the major drift line; 1,348 decisions held for human review. No-drift stream: 0 holds, 0 village flags, 2 age-band advisories (never added to the hold list).
- Receipt signature OK; erasure log chain OK; numbers validator PASS (template writer).
- Shard retrain 0.28 s; parallel full retrain 1.50 s; serial full retrain 2.52 s (receipt). The serial figure is a one-at-a-time retrain of every shard; any ratio to it must say so.

## Benchmarks (named machines, never mixed)
- x86_64, 2 cores, cloud container (`docs/evidence/bench_snapshot_x86_64-2cores-cloud-rehearsal/`): train 8 shards 2.486 s with 1 worker, 1.559 s with 2; one shard retrain 0.330 s; parallel full retrain 1.457 s; serial 2.432 s; cold audit (includes building the shadow forests) 41.045 s with 1 worker, 22.008 s with 2. Two cores support no general speedup claim.
- WSL2 x86_64, 8 cores (`docs/evidence/bench_snapshot_wsl2_x86_64-8cores_rehearsal/`): train 8 shards 4.767 s with 1 worker, 1.721 s with 8; one shard 0.335 s; parallel full retrain 1.921 s; serial 2.929 s. One shard vs serial full retrain 8.73x: that ratio is one shard against retraining every shard one at a time, not an 8-core scaling result.
- Scheduling arithmetic (not measured): 8 shard tasks on a 3-core VM run in waves of 3, 3 and 2.

## Accuracy cost of sharding (`docs/evidence/shards_curve/results.csv`, x86_64 2-core container)
- Test AUROC: 1 shard 0.758, 4 shards 0.774, 8 shards 0.776, 16 shards 0.775. No accuracy cost observed on synthetic data.

## Audit gate, before erase (WSL2 x86_64, `docs/evidence/audit_gate_local.log`)
- Worst-shard calibrated attack AUC 0.988; whole-model calibrated 0.825 (interval 0.797 to 0.854), raw 0.613. Gate PASS.

## Known limitation (SC-13, `docs/evidence/sc13_diagnostics/summary.md`)
- After erase, the shard-level attack AUC lands at or below chance (never described as an exact chance value). Cause: a calibration asymmetry. Over 9 seeds: whole-model mean 0.494, shard-level calibrated mean 0.478, raw mean 0.499. Matched held-out-only pseudo-members reproduce the shift, so it is a calibration effect, not leftover membership signal.

## Optional components (x86_64 only, own venvs)
- ONNX export: parity with scikit-learn within 2.9e-07 on 1,000 probe rows (`docs/evidence/onnx_env.md`).
- Differential privacy (diffprivlib): raw attack AUC 0.996 for the non-DP forest vs 0.529 to 0.534 for DP forests at epsilon 0.5 to 10; test AUROC 0.758 vs 0.654 to 0.660 (`docs/evidence/dp_env.md`). A separate guarantee from unlearning.

## Planned (not run, do not present as done)
- Anything on the LinuxONE s390x VM: install, byte-order test, benchmarks, demo.
- IBM Granite via llama.cpp, IBM Z Deep Learning Compiler, Triton serving.
- Telum, Spyre and Secure Execution are not used and not claimed.

## Open for a person to fill in
- LinuxONE SSH line and README host; repository URL; DPDP clause and commencement dates (verify with counsel); Official Rules (deadline, timezone, team-size rule).
