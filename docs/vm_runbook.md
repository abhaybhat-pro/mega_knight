# VM runbook (LinuxONE s390x). Planned until a person runs it.

A person supplies the SSH line and the host. This repository does not contain them and must not guess them.

1. Connect with the SSH line you were given, then get the repository onto the VM
   (`git clone <repo-url>` once it is pushed, or copy `forgetme-aegis-checkpoint.zip` and unzip it).
2. `cd forgetme-aegis && bash scripts/vm_runbook.sh`
   It runs setup (`scripts/setup_s390x.sh`), a version and byte-order record, a Python 3.10 syntax check, the fast tests,
   the byte-order test, the contract check, the audit gate, the demo, the shards curve, the benchmarks with resource profile,
   the slow tests and the claims check. A failed step does not stop the run.
3. Read `docs/evidence/vm_run_<utc>/SUMMARY.md`. The first lines say whether the host was s390x.
   Benchmarks land in `docs/evidence/bench_snapshot_s390x-<N>cores/`.
4. Copy `docs/evidence/` back and commit it. Only then update `docs/ibm_status.md` (Planned to Measured) and
   `docs/submission/facts.md`, using numbers from those files and nowhere else.

If a step fails, keep its log, note the failure in `docs/ibm_status.md`, and do not retry in a loop.
Time-box any package that will not install (`LIB_TIMEOUT`, default 15 minutes).

Do not export the HMAC key or write it to a file; the runbook creates a throwaway one for the run.
