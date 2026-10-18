# SPEC_CHANGES

Proposals are below. Lead decisions are in the table that follows.
Frozen thresholds are not in this list. If a threshold is missed, use the 8.5 ladder.

## Lead decisions (18 Oct 2026, 02:10 IST)

These calls are for the owners named in each item. Feature files stay as delivered until that owner edits them. No threshold number changes.

| Item | Decision |
| --- | --- |
| SC-1 | Approved as a bugfix. the erase module passes `data_dir=data_dir` into both `mia.run_audit` calls (queue.py lines 420 and 459). the audit module's default stays `"data"`. |
| SC-2 | Approved. the shard layer's key wins. `_update_manifest` sets `shards[k]["model_hash"]` to the new fingerprint. Do not also write `hash` on that entry. Nothing in the tree reads `shards[k]["hash"]`. |
| SC-3 | the audit module reads `THRESHOLDS["n_bootstrap"]`. Gate floors stay 0.90 and 0.62 as the constants already used in `scripts/audit_gate.py` and `tests/test_audit.py`. Do not add `gate_shard_auc` or `gate_ensemble_auc` to `THRESHOLDS`. |
| SC-4 | Approved. When `data/train_current.csv` exists, the full-retrain timing uses it (`load_train_df`). Otherwise it uses `train.csv`. Owner: the bench/report module. |
| SC-5 | Approved. The Benchmarks page reads `bench/shards_curve/results.csv`. `bench/results.csv` stays the FR-10 benchmark file. |
| SC-6 | Approved, and it does not block Gate 1. the audit module adds `n_workers: int = -1` to `run_audit` and passes it into `get_shadows` as `n_jobs`. `get_shadows` already takes `n_jobs`. Until that parameter exists, the bench/report module keeps the audit-scaling stand-in and says so in `results_meta.json`. |
| SC-7 | The band "absolute distance of AUC from 0.5 at most 0.05" applies to the erase of every member canary (all shards, the case `test_after_erase` asserts). A single-shard erase keeps the audit module's current test: shard AUC must fall by more than 0.30, and the ensemble number is printed. The demo still erases one shard. The slide for that erase shows the drop. It does not claim the 0.05 band. |
| SC-8 | Keep `models/person_shard_map.json`. It is an operational index (person id to shard), recomputable with `shard_of`, and it is not a receipt field and not a log field. |
| SC-9 | Window length is 2 batches (1000 rows). See the note under SC-9 for the count. Age-band drops are shown and do not add anyone to the hold list. The hold list is person ids from villages at or above `psi_major` (0.25), stored in the Streamlit session only. |
| SC-10 | Approved as an optional receipt field. `guard.flagged_subgroups` is `{label: drop}` beside the four guard fields. the Guard/UI module and the bench/report module accept the key when it is present and accept a receipt that omits it. A printed drop is taken from the receipt. Labels only: no person ids. |

Windows `os.fsync` on the read-only descriptor in `_fsync_path` is not a spec change. Re-run `test_scratch_full_retrain_is_timed_but_never_swapped_in` on the LinuxONE VM before editing that function.

There is no LinuxONE shell on this machine. `~/.ssh` does not exist, and the README host is still `<TODO>`. The s390x run waits on the SSH command from the LinuxONE Community Cloud reservation.

## SC-1. Forward `data_dir` into `run_audit` (bug, not a spec change)

- What: `erase()` loads held-out canaries from its `data_dir` argument, then calls `run_audit(ss, members, held, shard_id=k)` with no `data_dir`.
- Why it breaks: the audit module's optional parameter defaults to `"data"`. A temp directory, or any cwd that is not the repo root, raises `FileNotFoundError: data/heldout_canaries.csv` before the shard is retrained.
- Affected: `src/erase/queue.py` lines 420 and 459. Owner: the erase module.
- Recommended fix: `mia.run_audit(ss, m, held_df, shard_id=k, data_dir=data_dir)`.
- Alternative: the audit module changes the default. That is worse. The caller knows the directory.

## SC-2. Manifest field name

- What: the shard layer writes `manifest["shards"][k]["model_hash"]` in `src/model/shards.py` (`build_manifest`). the erase module writes `shards[k]["hash"]` in `_update_manifest` and does not update `model_hash`.
- Why: after a real erase the file on disk is the new forest and the manifest's `model_hash` is still the old one.
- Affected: `src/erase/queue.py`. Owner: the erase module. the shard layer's layout wins.
- Recommended fix: set `ent["model_hash"]` to the new fingerprint. Keep `hash` as well only if something already reads it.
- This was not reached in the runtime test because SC-1 crashes first.

## SC-3. THRESHOLDS key names (the shard layer's names are the contract)

the shard layer's keys, now in the merged `src/contracts.py`:

| Value | Key to import |
| --- | --- |
| 0.10 PSI warning | `psi_moderate` |
| 0.25 PSI flag | `psi_major` |
| 200 | `min_subgroup_size` |
| 0.10 confidence drop | `confidence_drop` |
| 0.03 subgroup AUROC drop | `subgroup_auroc_drop` |
| 0.5 referral | `referral_threshold` |
| 1000 bootstraps | `n_bootstrap` |
| 16 shadows | `n_shadow_models` |
| 10 bins | `n_bins` |
| 1000 probe rows | `probe_rows` |
| 0.01 FPR | `fpr_target` |

the audit module still reads `bootstrap_resamples`, `gate_shard_auc`, and `gate_ensemble_auc`, with fallbacks 1000, 0.90, and 0.62. The numbers match. The names do not.
Recommended: the audit module reads `n_bootstrap`. Gate floors can stay as fallbacks, or the shard layer can add `gate_shard_auc` and `gate_ensemble_auc` as the same frozen numbers. Adding keys does not change a threshold.

the Guard/UI module's stub used `min_subgroup_n`, `conf_drop`, `referral`. That stub was not merged. When `monitor.py` is written it must use the table above.

## SC-4. Benchmark training file

- What: `bench/run_benchmarks.py` reads `train.csv` only.
- Why: after an erase the live set is `data/train_current.csv`. A full retrain timed on `train.csv` puts the erased people back.
- Recommended: if `train_current.csv` exists, the benchmark and any "full retrain" timing use it. Owner: the bench/report module. the erase module already exposes `load_train_df`.

## SC-5. Curve CSV path

- the shard layer writes `bench/shards_curve/results.csv` (columns `n_shards`, `rows_per_shard`, `test_auroc`, `auroc_delta_vs_fewest`).
- the Guard/UI module's Benchmarks page reads `bench/shards_curve.csv`.
- Recommended: the Guard/UI module reads `bench/shards_curve/results.csv`. Do not also write `bench/results.csv` for the curve. That path belongs to FR-10.

## SC-6. `run_audit` worker argument

- the bench/report module's audit-scaling benchmark uses a stand-in until `run_audit` accepts optional `n_workers`.
- Recommended: the audit module adds `n_workers: int = -1` and passes it to `get_shadows`. This is an extra optional parameter, which the spec allows.
- Not a reason to block the audit merge.

## SC-7. Single-shard after-erase band (team decision, threshold unchanged)

the audit module reported that `|AUC - 0.5| <= 0.05` on one shard (~50 members vs 400 held-out) fails about 20% of the time for a perfect erase. Their test asserts that band only after all 400 member canaries are removed. The single-shard number is printed, not asserted.

No threshold change is proposed. The human lead decides whether the acceptance row applies to one shard or to the all-canary erase. Until then the frozen test stays as the audit module wrote it.

## SC-8. `person_shard_map.json`

the shard layer writes `models/person_shard_map.json` (person id to shard). Shard assignment can be recomputed with `shard_of`. Rule 6 prefers counts and metrics in persisted state.
Recommended decision for the lead: stop writing the map, or accept it as non-clinical operational state. Merge lead will not delete it.

## SC-9. Guard window and age-band hold list

Decided. the Guard/UI module writes `monitor.py` to this call. `min_subgroup_size` stays 200. `psi_moderate` stays 0.10. `psi_major` stays 0.25. The only knob that moves is the window length.

Village 3 is 25% of ordinary rows (`_VILLAGE_P` in `src/data/generate.py`). A 500-row batch therefore carries about 125 rows from that village. One batch stays under 200. Two batches are about 250, which clears 200. The comparison window is those 2 batches (1000 rows). A village that still has fewer than 200 rows in that window is omitted, not flagged. The two rare villages are about 3% each and will not reach 200 inside this window. That is the frozen minimum at work.

Age-band AUROC drops are displayed. They do not add person ids to the hold list. The hold list is person ids from villages whose PSI is at or above 0.25, kept in `st.session_state` only.

## SC-10. `flagged_subgroups` on the receipt

the erase module adds optional `guard.flagged_subgroups` (`{label: drop}`) beside the four 8.4 guard fields. It is additive. the Guard/UI module and the bench/report module should tolerate it. `validate_numbers` already walks nested numbers, so a report that prints those drops must take them from the receipt.

## SC-13 (APPROVED by the lead): one-sided subset-level after-erase assertions

Status: APPROVED with an exact scope. Only the two subset-level assertions in `tests/test_audit.py::test_after_erase`
changed: the shard-level CI of the pre-registered shard (3) and the ensemble CI from the single-shard audit.
Before: `lo <= 0.5 <= hi`. After: `lo <= 0.5` (the attack is not significantly better than chance).
The all-400-member ensemble assertions (AUC within 0.05 of 0.5, CI contains 0.5) and every before-erase floor are unchanged.
A power check was added: the same one-sided test must FAIL on the before-erase audit of the same shard and ensemble subset.
The informational two-sided coverage print is kept.
Reproduced value: shard-3 attack AUC 0.4635 with the CI not containing 0.5 on the low side.

Measured (evidence in `docs/evidence/sc13_diagnostics/`, 9 seeds, x86_64 rehearsal):
- Ensemble AUC after a full erase averages 0.494 over the 9 seeds, so the ensemble claim holds.
- Mean calibrated shard AUC averages 0.478, against a raw (uncalibrated) shard AUC near 0.499 in the earlier run.
- Matching held-out-only pseudo-members gives 0.477, so the shift comes from calibration, not from leftover membership signal.
- Label imbalance does not explain it (label-matched 0.489 vs 0.488). Generator parity does not explain it (feature KS p > 0.2).

Cause: shard-level calibration scores each held-out canary with the shadows of the shard it hashes to,
which makes the in and out calibration unequal. The result is a small downward bias of the shard AUC.

The audit calibration itself is unchanged. Disclosure wording (README limitations, facts file, ibm_status):
post-erase shard-level attack AUC lands "at or below chance"; never "exactly 0.5".

## SC-9 (lead: keep as implemented)

Guard hold list = persons in villages with max feature PSI >= 0.25. Confidence drops are warnings; age-band drops are advisory.
The original spec also held on a 0.10 confidence drop. The lead chose PSI-only.

## SC-11 / SC-12 (recorded)

- SC-11: plotly is allowed in `app/` only. Nothing under `src/` imports it, and the UI falls back to Streamlit native charts when it is absent.
- SC-12: the headline `full_retrain_seconds` is the PARALLEL retrain on all available cores. The receipt also carries
  `full_retrain_serial_seconds` and `speedup_vs_serial`. "Speedup" means parallel baseline / shard retrain.
  Never state "8x" without saying it is against a serial retrain. No speedup measured on a 1-2 core container is a general result.

## make_data (lead-approved)

`make_data` returns `(df, {"member": [...], "heldout": [...]})`; documented in the facade docstring and README.
