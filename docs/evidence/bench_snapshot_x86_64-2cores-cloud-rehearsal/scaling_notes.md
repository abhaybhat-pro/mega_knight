# Where scaling flattens (Amdahl)

Measured on a machine with 2 core(s) (`nproc`), x86_64, Python 3.13.16, byte order little, 20000 training rows, median of 3 runs. Everything below is computed from bench/results.csv.

## train_scaling (8 shards)

| workers | seconds | speedup vs 1 worker |
|---|---|---|
| 1 | 2.486 | 1.00x |
| 2 | 1.559 | 1.59x |
| 4 | 1.603 | 1.55x |
| 8 | 1.907 | 1.30x |

Estimated serial fraction (Amdahl, workers up to 2): 0.254, which caps speedup at about 3.9x.
Eight shard tasks on 2 cores fill 100% of the core-slots (the last wave can leave a core idle). That ceiling is the train_all limit on this machine. x86_64 rehearsal, not s390x.

## audit_scaling (8 shards)
Source: real src.audit.mia.run_audit. Cold cache: True. Repeats per row: 1. Worker counts above the core count were not timed.

| workers | seconds | speedup vs 1 worker |
|---|---|---|
| 1 | 41.045 | 1.00x |
| 2 | 22.008 | 1.87x |

Estimated serial fraction (Amdahl, workers up to 2): 0.072, which caps speedup at about 13.8x.
Eight shard tasks on 2 cores fill 100% of the core-slots (the last wave can leave a core idle). That ceiling is the train_all limit on this machine. x86_64 rehearsal, not s390x.

Causes to check (not measured here): process start-up and data transfer to workers, the slowest single shard (a pool cannot finish before it), memory bandwidth, and the cores of the VM being shared.
