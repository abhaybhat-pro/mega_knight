"""Short names over the frozen 8.3 functions. The stubs in contracts.py stay as they are.

Decision (lead-approved): ``make_data`` returns ``(df, {"member": [...], "heldout": [...]})``.
The second element, ``canary_ids``, holds lists of
person ids whose ``canary_type`` column says so ("member" ids come from ``df``, "heldout" ids from the
held-out canary file). They are not chosen by id range.
``n`` defaults to ``N_PATIENTS``. The generator always writes that many rows;
a smaller ``n`` keeps every member canary and fills the rest with ordinary rows.
"""
from __future__ import annotations

from src.runtime.resources import configure_threads

configure_threads()

import numpy as np
import pandas as pd

from src.contracts import (
    CANARY_COL, DATA_DIR, HELDOUT_CSV, ID_COL, LOG_PATH, N_PATIENTS, N_SHARDS, SEED, TRAIN_CSV,
)
from src.data.generate import generate, read
from src.erase.queue import erase as erase_flow
from src.erase.queue import load_train_df
from src.guard.monitor import compare_subgroup_auroc, fit_baseline, score_batch
from src.model.shards import load, predict, shard_of, train_all
from src.audit.mia import run_audit
from src.report.receipt import (
    build_receipt, verify_log, verify_receipt_log_linkage,
)


def make_data(seed: int = SEED, n: int = N_PATIENTS):
    generate(seed=seed, out_dir=DATA_DIR)
    df = read(TRAIN_CSV, DATA_DIR)
    held = read(HELDOUT_CSV, DATA_DIR)
    if n < len(df):
        members = df[df[CANARY_COL] == "member"]
        ordinary = df[df[CANARY_COL] != "member"]
        if n < len(members):
            raise ValueError("n is smaller than the member-canary count")
        ordinary = ordinary.sample(n=n - len(members), random_state=seed)
        df = (pd.concat([ordinary, members], ignore_index=True)
              .sort_values(ID_COL, kind="mergesort").reset_index(drop=True))
    canary_ids = {
        "member": [int(v) for v in df.loc[df[CANARY_COL] == "member", ID_COL].tolist()],
        "heldout": [int(v) for v in held.loc[held[CANARY_COL] == "heldout", ID_COL].tolist()],
    }
    return df, canary_ids


def train_shards(df: pd.DataFrame, n_shards: int = N_SHARDS, seed: int = SEED):
    return train_all(df, n_shards=n_shards, seed=seed)


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    return np.asarray(predict(load(), X)["proba"])


def erase(person_ids, request_id, guard_summary: dict | None = None) -> dict:
    """Load the served model and the current training frame, erase, return counts.

    ``guard_summary`` is optional counts from ``src.guard.monitor.stream_counts``.
    """
    receipt = erase_flow(person_ids, request_id, load(), load_train_df(),
                         guard_summary=guard_summary)
    return {
        "timings": {
            "full_retrain_seconds": receipt.get("full_retrain_seconds"),
            "shard_retrain_seconds": receipt.get("shard_retrain_seconds"),
            "speedup": receipt.get("speedup"),
        },
        "hashes": {
            "before": receipt.get("model_hash_before"),
            "after": receipt.get("model_hash_after"),
        },
        "rows_removed": receipt.get("rows_removed"),
        "receipt": receipt,
    }


def _view(block):
    if not isinstance(block, dict):
        return None
    return {
        "auc": block.get("attack_auc"),
        "tpr_at_1fpr": block.get("tpr_at_1pct_fpr"),
        "canary_pct": block.get("canary_percentile_median"),
        "ci": block.get("auc_ci95"),
    }


def audit(models, members, nonmembers, shard_id=None) -> dict:
    raw = run_audit(models, members, nonmembers, shard_id)
    ensemble = _view(raw.get("ensemble"))
    shard = _view(raw.get("shard"))
    top = shard if shard_id is not None and shard is not None else ensemble
    top = top or {"auc": None, "tpr_at_1fpr": None, "canary_pct": None, "ci": None}
    return {**top, "ensemble": ensemble, "shard": shard}


def guard_check(batch) -> dict:
    ss = load()
    base = fit_baseline(load_train_df(), ss=ss)
    report = score_batch(ss, batch, base)
    return {"psi": report.psi, "entropy": report.entropy, "held_ids": list(report.held_ids)}


def subgroup_delta(before, after) -> dict:
    return compare_subgroup_auroc(before, after)


def make_receipt(**fields) -> dict:
    return build_receipt(**fields)


def verify_chain(path: str | None = None, receipt: dict | None = None) -> bool:
    """True when the log chain is intact and, if a receipt is given, it links."""
    log_path = path or LOG_PATH
    ok, _bad = verify_log(log_path)
    if not ok:
        return False
    if receipt is None:
        return True
    linked, _why = verify_receipt_log_linkage(receipt, log_path)
    return bool(linked)
