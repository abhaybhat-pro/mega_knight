# DEMO_RUNBOOK

Not ready for a live demo. Do not record a claim from this file.

Blocked before any rehearsal:

1. `src/guard/monitor.py` does not exist. `bash scripts/run_demo.sh` calls `need("src.guard.monitor")` and exits 3.
2. `erase()` does not pass `data_dir` into `run_audit` (SPEC_CHANGES SC-1). A member-canary erase raises `FileNotFoundError` unless the process cwd already has `data/heldout_canaries.csv`.
3. No run on the LinuxONE s390x VM. Laptop results are not the demo.

`--no-guard` skips step 4 only. It does not fix SC-1, and it is not the Gate 3 demo.

## Intended 3-minute order (PRD section 2), after the blockers are gone

Run from a fresh clone on the VM, after `bash scripts/setup_s390x.sh`:

```bash
export FORGETME_HMAC_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
bash scripts/run_demo.sh
```

Expected shape of the output, not a transcript from this machine:

1. Baseline audit prints shard and ensemble attack AUC.
2. One shard's member canaries are erased. The other shard files stay.
3. The receipt's after-audit is lower than the before-audit on that shard.
4. Guard reports village 3 and does not flag the no-drift stream.
5. Receipt path is the last line. Signature and log chain both OK.
6. Tamper: `python3 scripts/tamper_demo.py` shows an invalid signature or a broken link.
7. Report writer is `template` or `granite`, and `validate_numbers` passes.

Run the same command twice. Attack numbers must match. If they do not, stop and do not present the run.

## Fallback

`docs/demo_recording.md` describes a backup recording. No backup recording was verified in this pass. A recording made on a laptop is not a substitute for the VM note in the deck.

## Honesty lines that have to be said out loud

- The audit is evidence from this attack, not a proof that the person is forgotten.
- Not legal advice. The status of model weights under the DPDP Act is unsettled. Dates in the report are marked to be verified.
- Differential privacy and shard unlearning are different guarantees.
- Granite, ONNX, Triton, and the IBM Z Deep Learning Compiler are Planned until a log says they ran.
- Do not say Telum, Spyre, or Secure Execution.
