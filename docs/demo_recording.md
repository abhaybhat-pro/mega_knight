# Backup demo recording (do this on the VM, once the real modules have landed)

Goal: if the live VM or network fails on stage, play a recording of a real run. Record only
real runs; never edit numbers.

## Terminal run (about 1 minute)
    cd forgetme-aegis
    script -q -t docs/demo_session.timing -c "bash scripts/run_demo.sh" docs/demo_session.typescript
    # replay later:  scriptreplay docs/demo_session.timing docs/demo_session.typescript
Also keep a plain copy: `bash scripts/run_demo.sh 2>&1 | tee docs/demo_run.txt`

## Screen recording of the Streamlit UI (about 90 seconds)
1. `streamlit run <the UI entry file>` (path set by the UI owner), open it in a browser, start the recorder
   (OS recorder, or OBS) before the first click.
2. Show in order: the guard drift stream flagging village 3, one erase request, the receipt,
   the before/after audit, the report (point at the first line saying which writer made it),
   the verification result.
3. Narrate the Done / Measured / Planned table from `docs/ibm_status.md` for 15 seconds. Say
   plainly what is Planned.
4. Save as `docs/demo_backup.mp4` (do not commit large video to git; keep it on the team drive).

## Checklist before the talk
- [ ] Fresh clone run exits 0 and prints the receipt path
- [ ] Two runs give identical `receipts/demo_attack_numbers.json`
- [ ] Recording is from a run on the VM (the first line of the run shows `s390x`)
