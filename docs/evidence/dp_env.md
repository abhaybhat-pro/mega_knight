# Differential privacy attempt (one attempt, dedicated venv, x86_64 cloud container, not s390x)

Venv outside the repo: `uv venv --python 3.12`, then diffprivlib 0.6.6 with scikit-learn 1.5.2 (the 2026-10-17 attempt with the newest scikit-learn failed on `sklearn.tree._tree.DOUBLE`, see `optional_install.md`; pinning scikit-learn 1.5.2 fixed the import).
Install and test log: `docs/evidence/dp_attempt_x86_64.log` (`tests/test_dp.py`: 2 passed). Table: `docs/evidence/dp_table_x86_64.txt`.

Result on this machine: a single non-DP forest has raw attack AUC 0.996 and test AUROC 0.758. Diffprivlib forests at epsilon 0.5, 1, 5 and 10 have raw attack AUC 0.529 to 0.534 (95% intervals reach down to 0.488 to 0.495) and test AUROC 0.654 to 0.660, a cost of about 0.10 AUROC. Part of that cost is the depth-limited DP forest, not only noise (see `src/model/dp.py`).

Differential privacy is a prior limit on one record's influence. It is a separate guarantee from shard unlearning, which is exact erasure after the fact. Neither is presented as evidence for the other. The diffprivlib feature bounds come from the training data, which a strict accounting would count against the budget.
