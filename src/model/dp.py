"""Differential-privacy comparison (FR-16). diffprivlib is allowed ONLY in this file.

DP is a SEPARATE guarantee from unlearning:
  * DP (this file)  = a PRIOR limit on how much any single record can influence the model.
  * Unlearning      = POST-HOC exact erasure by retraining the affected shard.
Neither implies the other. Do not present one as evidence for the other.

Comparison design (like-for-like): a single non-DP sklearn forest (same RF settings, same data)
is the baseline; DP forests (diffprivlib) at several epsilons are compared against it.
Attack AUC here = RAW logit-true-score AUC of the 400 member canaries vs the 400 held-out canaries
(no shadow calibration for DP models: out of scope and costly) with a bootstrap 95% CI.

Caveats (stated, not hidden):
  * diffprivlib needs feature bounds. We take them from the training data, which itself spends
    privacy budget in a strict DP accounting. Use public/domain bounds in a real deployment.
  * diffprivlib's RF is depth-limited (max_depth=5 default) and uses random splits, so part of the
    test-AUROC drop is model-capacity, not only the noise. Both effects are real costs of DP here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.metrics import roc_auc_score

from src.contracts import FEATURES, SEED
from src.audit.mia import auc, bootstrap, logit_true, proba_pos

DEFAULT_EPSILONS = (0.5, 1.0, 5.0, 10.0)


def _bounds(df: pd.DataFrame):
    X = df[FEATURES]
    return (X.min().values.astype(float), X.max().values.astype(float))


def _attack(model, members, heldout, seed_off):
    m = logit_true(proba_pos(model, members[FEATURES]), members["needs_referral"].values)
    n = logit_true(proba_pos(model, heldout[FEATURES]), heldout["needs_referral"].values)
    ci, _ = bootstrap(m, n, np.random.RandomState(SEED + seed_off))
    return auc(m, n), ci


def train_baseline(train_df: pd.DataFrame, n_estimators: int = 100) -> SkRF:
    d = train_df.sort_values("person_id")
    return SkRF(n_estimators=n_estimators, max_depth=None, min_samples_leaf=1, bootstrap=True,
                n_jobs=1, random_state=SEED).fit(d[FEATURES], d["needs_referral"])


def train_dp(train_df: pd.DataFrame, epsilon: float, n_estimators: int = 100):
    from diffprivlib.models import RandomForestClassifier as DPRF   # imported lazily: optional dependency
    d = train_df.sort_values("person_id")
    return DPRF(n_estimators=n_estimators, epsilon=float(epsilon), bounds=_bounds(train_df),
                classes=[0, 1], random_state=SEED).fit(d[FEATURES], d["needs_referral"])


def run_dp_experiment(train_df, test_df, heldout_df, epsilons=DEFAULT_EPSILONS,
                      n_estimators: int = 100, out_csv: str | None = None) -> pd.DataFrame:
    """Return a table: model, epsilon, attack_auc_raw, ci_lo, ci_hi, test_auroc (+ deltas vs baseline)."""
    members = train_df[train_df["canary_type"] == "member"]
    y_te = test_df["needs_referral"].values

    def row(name, eps, model, off):
        a, ci = _attack(model, members, heldout_df, off)
        return {"model": name, "epsilon": eps, "attack_auc_raw": a, "attack_ci_lo": ci[0], "attack_ci_hi": ci[1],
                "test_auroc": float(roc_auc_score(y_te, proba_pos(model, test_df[FEATURES])))}

    rows = [row("baseline_rf_single", None, train_baseline(train_df, n_estimators), 31)]
    for i, e in enumerate(epsilons):
        rows.append(row("dp_rf", float(e), train_dp(train_df, e, n_estimators), 32 + i))
    df = pd.DataFrame(rows)
    b = df.iloc[0]
    df["d_attack_auc"] = df["attack_auc_raw"] - b["attack_auc_raw"]
    df["d_test_auroc"] = df["test_auroc"] - b["test_auroc"]
    if out_csv:
        df.to_csv(out_csv, index=False)          # scores/metrics only, no patient features
    return df


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    he = pd.read_csv(os.path.join(a.data_dir, "heldout_canaries.csv"))
    print(run_dp_experiment(tr, te, he, out_csv=a.out).round(4).to_string(index=False))
    print("\nDP (prior influence limit) and unlearning (post-hoc erasure) are SEPARATE guarantees.")
