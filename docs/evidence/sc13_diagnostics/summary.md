# SC-13 diagnostics: after-erase audit, 9 seeds (x86_64, 2-core rehearsal, not s390x)

Produced by `python scripts/diagnose_after_erase.py --seed S --n-splits 100 --json ...`. Raw JSON beside this file.

| seed | ensemble AUC after erase | mean shard AUC | held-out-only matched | matched both sides |
|---|---|---|---|---|
| 42 | 0.488 | 0.446 | 0.478 | 0.464 |
| 1 | 0.496 | 0.460 | 0.490 | 0.464 |
| 2 | 0.488 | 0.477 | 0.466 | 0.504 |
| 3 | 0.488 | 0.475 | 0.483 | 0.491 |
| 4 | 0.468 | 0.503 | 0.473 | 0.532 |
| 5 | 0.445 | 0.455 | 0.473 | 0.474 |
| 6 | 0.511 | 0.507 | 0.489 | 0.518 |
| 7 | 0.527 | 0.478 | 0.479 | 0.497 |
| 8 | 0.537 | 0.499 | 0.459 | 0.535 |
| mean | 0.494 | 0.478 | 0.477 | 0.498 |
