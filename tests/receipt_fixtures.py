"""Shared fixtures for the erase/receipt/log tests (receipt/log/erase/queue)."""
from __future__ import annotations

KEY = b"test-key-not-a-secret"


def view(auc=0.98):
    return {"attack_auc": auc, "auc_ci95": [auc - 0.04, min(1.0, auc + 0.01)],
            "tpr_at_1pct_fpr": 0.61, "canary_percentile_median": 0.97}


def pre_receipt_fields(**over):
    base = dict(
        request_id="REQ-0001",
        timestamp_utc="2026-10-17T16:10:12Z",
        persons_erased=6,
        shard_ids=[3],
        rows_removed=6,
        model_hash_before={"3": "a" * 64},
        model_hash_after={"3": "b" * 64},
        full_retrain_seconds=12.5,
        shard_retrain_seconds=1.6,
        speedup=7.8125,
        test_auroc_before=0.8123,
        test_auroc_after=0.8101,
        audit_before={"ensemble": view(0.71), "shard": view(0.98)},
        audit_after={"ensemble": view(0.50), "shard": view(0.54)},
        guard={"worst_subgroup": "village_3", "max_subgroup_auroc_drop": 0.011,
               "drift_flags": None, "held_decisions": None},
    )
    base.update(over)
    return base


def make_pre_receipt(**over):
    from src.report.receipt import build_receipt
    return build_receipt(**pre_receipt_fields(**over))
