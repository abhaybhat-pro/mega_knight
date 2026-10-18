"""Secondary MIA check check: 500 ordinary TRAINING rows vs 500 TEST rows.

Raw-score only, by construction: ordinary rows are in EVERY shadow, so there is no per-example
'out' distribution to calibrate against. Expected to be much weaker than the canary attack.
Ensemble level uses the deployed ensemble; shard level scores each training row with its OWN
shard model and each test row with the same shard model it is paired to (a random one, seeded).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from src.contracts import SEED, FEATURES
from src.model.shards import shard_of
from src.audit.mia import (auc, bootstrap, tpr_at_fpr, logit_true, proba_pos, target_scores,
                           canary_percentile_median)


def run_secondary_audit(ss, train_df: pd.DataFrame, test_df: pd.DataFrame, n: int = 500) -> dict:
    ordinary = train_df[train_df["canary_type"] == "none"]
    tr = ordinary.sample(n=min(n, len(ordinary)), random_state=SEED).sort_values("person_id")
    te = test_df.sample(n=min(n, len(test_df)), random_state=SEED).sort_values("person_id")
    out = {}
    # ensemble level
    m, nn = target_scores(ss, tr, "ensemble"), target_scores(ss, te, "ensemble")
    out["ensemble"] = _blk(m, nn, 11)
    # shard level: own shard for train rows; seeded random shard for test rows
    rng = np.random.RandomState(SEED + 5)
    tr_sh = np.array([shard_of(int(p), ss.n_shards) for p in tr["person_id"]])
    te_sh = rng.randint(0, ss.n_shards, len(te))
    ms = _by_shard(ss, tr, tr_sh)
    ns = _by_shard(ss, te, te_sh)
    out["shard"] = _blk(ms, ns, 12)
    out["note"] = "raw scores only (no shadow calibration possible for ordinary rows); weaker than canaries by design"
    return out


def _by_shard(ss, df, shards):
    s = np.empty(len(df))
    for k in range(ss.n_shards):
        idx = np.where(shards == k)[0]
        if len(idx):
            s[idx] = target_scores(ss, df.iloc[idx], "shard", k)
    return s


def _blk(m, n, off):
    rng = np.random.RandomState(SEED + off)
    ci, tci = bootstrap(m, n, rng)
    return {"attack_auc_raw": auc(m, n), "auc_ci95_raw": ci, "tpr_at_1pct_fpr_raw": tpr_at_fpr(m, n),
            "tpr_ci95": tci, "canary_percentile_median": canary_percentile_median(m, n),
            "n_members": int(len(m)), "n_nonmembers": int(len(n))}
