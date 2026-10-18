# Stale delivery folders: comparison with the canonical project

Compared on 18 Oct 2026 before archiving, against the canonical folder exactly as delivered (before any change).
Method: SHA-256 of every file in each stale folder and in `forgetme-aegis-integrated.zip` against the file at the same
relative path in the canonical project (the seven loose files in `files/` were matched by name). "Older" was judged by
the original Windows modification times: every differing stale file had an earlier time than its canonical counterpart.

- Byte-identical to canonical: 51 loose files, 42 files inside the zip.
- Different from canonical and older (superseded): 35 loose files, 35 files inside the zip.
- Different and NOT older: none.
- No canonical counterpart: `app/mock.py` (a mock UI data helper; no file in the canonical project imports it).

Everything was archived to `archive_dev_drops.zip` outside the repository (not shipped).

Loose files that differ (stale size vs canonical size, bytes):

| Stale folder | File | Stale | Canonical |
|---|---|---|---|
| files | test_audit.py | 5957 | 6097 |
| files | audit_gate.py | 3712 | 3900 |
| files | test_dp.py | 1460 | 1409 |
| files | mia.py | 10951 | 13821 |
| forgetme_dev3 | src/contracts.py | 1251 | 7043 |
| forgetme_dev3 | src/report/receipt.py | 26699 | 29910 |
| forgetme_dev3 | src/erase/queue.py | 24324 | 27394 |
| forgetme_dev3 | tests/test_erase.py | 14880 | 15060 |
| forgetme_dev3 | tests/test_log.py | 9049 | 9095 |
| forgetme_dev4 (1)/forgetme | src/contracts.py | 770 | 7043 |
| forgetme_dev4 (1)/forgetme | app/streamlit_app.py | 740 | 1042 |
| forgetme_dev4 (1)/forgetme | app/pages/3_Erase.py | 1711 | 3047 |
| forgetme_dev4 (1)/forgetme | app/pages/7_Benchmarks.py | 977 | 1019 |
| forgetme_dev4 (1)/forgetme | app/pages/1_Train.py | 585 | 1414 |
| forgetme_dev4 (1)/forgetme | app/pages/4_Audit.py | 1761 | 1532 |
| forgetme_dev4 (1)/forgetme | app/pages/5_Guard.py | 1313 | 3039 |
| forgetme_dev4 (1)/forgetme | app/pages/2_Predict.py | 1975 | 1607 |
| forgetme_dev4 (1)/forgetme | app/pages/6_Receipts.py | 889 | 938 |
| forgetme_dev4 (1)/forgetme | scripts/smoke_ui.py | 1260 | 3237 |
| forgetme-aegis-dev1 (1)/forgetme-aegis | src/contracts.py | 6497 | 7043 |
| forgetme-aegis-dev1 (1)/forgetme-aegis | src/model/shards.py | 8920 | 9313 |
| forgetme-aegis-dev1 (1)/forgetme-aegis | src/data/generate.py | 7784 | 8029 |
| forgetme-aegis-dev5/forgetme-aegis | README.md | 5866 | 9387 |
| forgetme-aegis-dev5/forgetme-aegis | .gitignore | 213 | 385 |
| forgetme-aegis-dev5/forgetme-aegis | src/contracts.py | 6497 | 7043 |
| forgetme-aegis-dev5/forgetme-aegis | src/model/shards.py | 8920 | 9313 |
| forgetme-aegis-dev5/forgetme-aegis | src/report/narrate.py | 20919 | 21863 |
| forgetme-aegis-dev5/forgetme-aegis | src/data/generate.py | 7784 | 8029 |
| forgetme-aegis-dev5/forgetme-aegis | src/guard/__init__.py | 0 | 62 |
| forgetme-aegis-dev5/forgetme-aegis | scripts/run_demo.sh | 985 | 980 |
| forgetme-aegis-dev5/forgetme-aegis | scripts/demo_pipeline.py | 10386 | 10688 |
| forgetme-aegis-dev5/forgetme-aegis | tests/test_validator.py | 9326 | 9578 |
| forgetme-aegis-dev5/forgetme-aegis | tests/test_benchmarks.py | 4529 | 5094 |
| forgetme-aegis-dev5/forgetme-aegis | docs/ibm_status.md | 2589 | 3359 |
| forgetme-aegis-dev5/forgetme-aegis | bench/run_benchmarks.py | 14681 | 20437 |
