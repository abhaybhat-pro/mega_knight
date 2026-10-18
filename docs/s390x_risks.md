# s390x risks and what is (not) verified

Everything here is a risk or a check to run on the VM. Nothing below was run on s390x. Today's runs are
x86_64 rehearsals and are labelled that way wherever a number appears.

| Risk | Why it matters | Status | How to check on the VM |
|---|---|---|---|
| Big-endian byte order | The prediction fingerprint hashes float64 values cast explicitly to little-endian (`<f8`, `src/model/shards.py`), so the hash should not depend on the host byte order. That is a design intent: only little-endian x86_64 has run. | Planned | `pytest tests/test_shards.py -q`; `python -c "import sys;print(sys.byteorder)"` should print `big` |
| Pickled / joblib models across byte orders | A model file written on x86_64 may not load on s390x (and the reverse). | Planned | Never copy `models/` between machines. Train on the VM (`scripts/run_demo.sh` does) |
| No s390x wheel for a dependency | pip may fall back to a source build. | Planned | `scripts/setup_s390x.sh` tries wheels, then apt, then a time-boxed source build, and logs each to `docs/evidence/install_log.md` |
| Python version on the image | The VM image may ship Python 3.10. | Syntax checked only | `python scripts/check_py310_syntax.py` parses every file with the 3.10 grammar (0 problems on the rehearsal). That is a syntax check, not a run on 3.10 |
| Minimum library versions | `requirements.txt` states lower bounds: numpy 1.26.4, pandas 2.1.4, scikit-learn 1.3.2, scipy 1.11.4, joblib 1.3.2. | See the final verification folder (the `final_verification` folder under the evidence directory, written by `scripts/final_check`) | The pinned-minimum run covers generate, shards, predict, receipt, log, validator, guard and facade only. Audit and benchmark paths were not run on the minimum set |
| Fewer cores than the rehearsal | 8 shard retrains scheduled on 3 cores run in waves of 3, 3 and 2. | Arithmetic only | The ceiling on core use is 8/9 (about 89%), and the speedup of one parallel pass over a serial pass is at most 8/3 (about 2.7x). These are scheduling rules, not measurements. Measure with `bench/run_benchmarks.py --profile` |
| Speedup claims | A speedup measured on 1 to 2 cores says nothing general. | Rule | The headline `full_retrain_seconds` is the PARALLEL retrain. Say "against a serial retrain" whenever a ratio to the serial row is quoted |
| Memory | The shadow build holds forests for 16 shadow ensembles. | Planned | `--profile` records peak RSS and the minimum MemAvailable |
| Optional components | ONNX, differential privacy, Granite, Triton and the Z Deep Learning Compiler. | Planned unless `docs/evidence/` holds a log of a successful run | `docs/ibm_status.md` |
| Hardware features | Telum, Spyre and Secure Execution. | Not used, not claimed | n/a |

The numbers shown to an audience come only from the LinuxONE VM run (`scripts/vm_runbook.sh`).
