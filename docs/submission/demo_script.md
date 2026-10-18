# Demo script (about 3 minutes). Say only what `facts.md` supports.

1. Problem (15 s): a clinic must erase a patient on request. Retraining the whole model each time is slow. Synthetic data only.
2. Run `bash scripts/run_demo.sh` (or show the UI Erase page). Name the machine on screen: today an x86_64 rehearsal, the VM run is Planned.
3. Guard (30 s): the drift stream makes one village drift; the decisions in that village are held for human review. Nothing is refused automatically.
4. Erase (30 s): one request, one shard of eight retrained. Show the shard and parallel-retrain times from the receipt.
5. Did it forget? (45 s): audit before and after. Shard attack score falls from 0.988 to 0.457. Say "at or below chance", say "evidence from this attack, not a proof".
6. Receipt (30 s): signature OK, log chain OK; then run the tamper demo and show INVALID.
7. Limits (20 s): synthetic data, not legal advice, DPDP dates to be verified, shard-level calibration effect (SC-13), s390x run Planned.
