# Benchmark snapshot: WSL2 x86_64, 8 cores, rehearsal (the lead's machine)

Copied unchanged from `bench/` as delivered. Not s390x. Not the cloud container (2 cores): that has its own snapshot
`../bench_snapshot_x86_64-2cores-cloud-rehearsal/`. Do not mix the two. These files predate the cold-cache change to
`audit_scaling`, so their `audit_scaling` rows are WARM-cache scoring times and are not a scaling result.
