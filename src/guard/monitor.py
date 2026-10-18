"""Aegis-Z Guard (FR-8, FR-12).

NO AUTOMATED REFUSALS: the only outcomes are proceed, warn, and hold-for-review.
A low or drifted score is held for a person to look at. Nothing here refuses care.

SC-9 (implemented): the hold list is person ids from villages whose max feature
PSI is at or above psi_major, and only when that village has at least
min_subgroup_size rows in the 2-batch window. Age-band drops are shown and do
not add anyone to the hold list. A village under 200 rows is "insufficient data".

Open question for the lead: FR-8 also describes a hold when mean confidence
falls by confidence_drop (0.10). SC-9 holds on PSI only. Confidence-drop flags
are warnings on the BatchReport, not holds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.runtime.resources import configure_threads

configure_threads()

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.contracts import (
    AGE_BANDS, DATA_DIR, FEATURES, LABEL_COL, N_VILLAGES, TEST_CSV, THRESHOLDS,
)
from src.model.shards import predict

_NUMERIC = ("age", "bmi", "sbp", "dbp", "hba1c", "prior_visits")
_BINARY = ("sex", "smoker", "sym_fatigue", "sym_vision", "sym_thirst", "sym_chest_pain")
_PSI_EPS = 1e-4
_LABEL = re.compile(r"[^A-Za-z0-9_.:+\-]+")
_WINDOW = 2  # batches. 1000 rows at the frozen batch size of 500.


def _clean_label(text: str) -> str:
    return _LABEL.sub("_", str(text))[:40] or "x"


def flatten_subgroup_auroc(tree, prefix: str = "") -> dict[str, float]:
    """Turn a nested or already-flat AUROC dict into {label: float}."""
    out: dict[str, float] = {}
    if not isinstance(tree, dict):
        return out
    for key, value in tree.items():
        lab = _clean_label(f"{prefix}{key}")
        if isinstance(value, dict):
            out.update(flatten_subgroup_auroc(value, lab + "_"))
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                out[lab] = number
    return out


def compare_subgroup_auroc(before, after, threshold: float | None = None) -> dict[str, float]:
    """Labels whose AUROC fell by more than the frozen subgroup_auroc_drop."""
    if threshold is None:
        threshold = float(THRESHOLDS["subgroup_auroc_drop"])
    left = flatten_subgroup_auroc(before)
    right = flatten_subgroup_auroc(after)
    flagged: dict[str, float] = {}
    for label in sorted(set(left) & set(right)):
        drop = left[label] - right[label]
        if drop > threshold:
            flagged[label] = float(drop)
    return flagged


@dataclass
class Baseline:
    """Feature baselines from training rows. Confidence baselines from test.csv."""

    specs: dict = field(default_factory=dict)
    test_df: pd.DataFrame | None = None
    confidence_mean: float | None = None
    entropy_mean: float | None = None
    window: list = field(default_factory=list)
    min_n: int = 200
    psi_moderate: float = 0.10
    psi_major: float = 0.25
    confidence_drop: float = 0.10
    n_bins: int = 10


@dataclass
class BatchReport:
    subgroups: list
    held_ids: list
    psi: dict
    entropy: float | None
    outcome: str
    flags: list


def _edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    raw = np.quantile(np.asarray(values, dtype=float), np.linspace(0.0, 1.0, n_bins + 1))
    return np.unique(raw.astype(float))


def _digit_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) < 2:
        return np.zeros(0, dtype=float)
    idx = np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=False)
    return np.bincount(idx, minlength=len(edges) - 1).astype(float)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    if len(expected) == 0 or len(expected) != len(actual):
        return 0.0
    exp = np.asarray(expected, dtype=float) + _PSI_EPS
    act = np.asarray(actual, dtype=float) + _PSI_EPS
    exp = exp / exp.sum()
    act = act / act.sum()
    return float(np.sum((act - exp) * np.log(act / exp)))


def _group_table(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(label, kind, row mask) for villages 0-7 and the three age bands."""
    rows: list[tuple[str, str, np.ndarray]] = []
    if "village_code" not in df.columns:
        return rows
    village = df["village_code"].to_numpy()
    for code in range(N_VILLAGES):
        rows.append((f"village_{code}", "village", village == code))
    if "age" not in df.columns:
        return rows
    age = df["age"].to_numpy()
    bands = []
    for lo, hi in AGE_BANDS:
        if hi >= 200:
            bands.append((f"age_{lo}_plus", age >= lo))
        else:
            bands.append((f"age_{lo}_{hi}", (age >= lo) & (age <= hi)))
    for label, mask in bands:
        rows.append((label, "age_band", mask))
    return rows


def _feature_spec(values: pd.Series, n_bins: int) -> dict | None:
    name = values.name
    arr = values.to_numpy()
    if name in _BINARY:
        cats = np.array([0, 1], dtype=float)
        counts = np.array([(arr == 0).sum(), (arr == 1).sum()], dtype=float)
        return {"kind": "binary", "categories": cats, "expected": counts}
    if name in _NUMERIC:
        edges = _edges(arr, n_bins)
        if len(edges) < 2:
            return None
        return {"kind": "numeric", "edges": edges, "expected": _digit_counts(arr, edges)}
    return None


def _psi_against(spec: dict, values: np.ndarray) -> float:
    if spec["kind"] == "binary":
        actual = np.array([(values == 0).sum(), (values == 1).sum()], dtype=float)
        return _psi(spec["expected"], actual)
    return _psi(spec["expected"], _digit_counts(values, spec["edges"]))


def _load_test(data_dir: str) -> pd.DataFrame | None:
    path = f"{data_dir}/{TEST_CSV}"
    try:
        return pd.read_csv(path)
    except OSError:
        return None


def fit_baseline(train_df: pd.DataFrame, ss=None, data_dir: str | None = None,
                 test_df: pd.DataFrame | None = None) -> Baseline:
    """Per-subgroup feature baselines from ``train_df``.

    Mean confidence and entropy come from ``test.csv`` (or ``test_df``), never
    from the training rows: an unlimited-depth forest is over-confident on data
    it has already fit. The deployed ShardSet is applied on the first score
    when it is not passed in here.
    """
    base = Baseline(
        min_n=int(THRESHOLDS["min_subgroup_size"]),
        psi_moderate=float(THRESHOLDS["psi_moderate"]),
        psi_major=float(THRESHOLDS["psi_major"]),
        confidence_drop=float(THRESHOLDS["confidence_drop"]),
        n_bins=int(THRESHOLDS["n_bins"]),
    )
    specs: dict[str, dict] = {}
    for label, _kind, mask in _group_table(train_df):
        rows = train_df.loc[mask]
        per_feature = {}
        for col in _NUMERIC + _BINARY:
            if col not in rows.columns or len(rows) == 0:
                continue
            spec = _feature_spec(rows[col], base.n_bins)
            if spec is not None:
                per_feature[col] = spec
        specs[label] = per_feature
    base.specs = specs
    if test_df is None:
        base.test_df = _load_test(data_dir or DATA_DIR)
    else:
        base.test_df = test_df
    if ss is not None and base.test_df is not None and len(base.test_df):
        _fill_confidence(ss, base)
    return base


def _fill_confidence(ss, base: Baseline) -> None:
    if base.confidence_mean is not None or base.test_df is None or len(base.test_df) == 0:
        return
    pred = predict(ss, base.test_df)
    base.confidence_mean = float(np.mean(pred["confidence"]))
    base.entropy_mean = float(np.mean(pred["entropy"]))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p))


def subgroup_auroc(ss, test_df: pd.DataFrame) -> dict:
    """Overall, per-village and per-age-band AUROC. One-class or small groups are skipped."""
    pred = predict(ss, test_df)
    y = test_df[LABEL_COL].to_numpy()
    p = np.asarray(pred["proba"], dtype=float)
    min_n = int(THRESHOLDS["min_subgroup_size"])
    out: dict = {"overall": None, "village": {}, "age_band": {}}
    overall = _auc(y, p)
    if overall is not None and len(y) >= min_n:
        out["overall"] = overall
    for label, kind, mask in _group_table(test_df):
        if int(mask.sum()) < min_n:
            continue
        score = _auc(y[mask], p[mask])
        if score is None:
            continue
        if kind == "village":
            out["village"][label.replace("village_", "")] = score
        else:
            out["age_band"][label.replace("age_", "")] = score
    return out


def public_counts(report: BatchReport) -> dict:
    """Counts and metrics only. Hold ids stay in memory."""
    return {
        "outcome": report.outcome,
        "entropy": report.entropy,
        "psi": report.psi,
        "n_held": len(report.held_ids),
        "subgroups": [
            {
                "label": row["label"],
                "kind": row["kind"],
                "n": row["n"],
                "mean_confidence": row["mean_confidence"],
                "mean_entropy": row["mean_entropy"],
                "max_psi": row["max_psi"],
                "flagged": row["flagged"],
                "reason": row["reason"],
                "status": row["status"],
            }
            for row in report.subgroups
        ],
    }


def _field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def stream_counts(reports) -> dict:
    """Counts only, safe to put on an erasure receipt (guard.drift_flags, guard.held_decisions).

    ``drift_flags``      number of distinct villages that reached hold-for-review (PSI at or above
                         the major line, at least min_subgroup_size rows) in any batch of the stream.
    ``held_decisions``   number of distinct person ids held for human review across the stream.
                         The 2-batch window overlaps, so ids are de-duplicated before counting.

    The ids themselves stay in memory inside this call and are never returned. Age-band results
    and moderate-band warnings are advisories: they are counted by ``stream_advisories``, not here.
    """
    villages: set[str] = set()
    held: set[int] = set()
    for report in reports or []:
        for row in _field(report, "subgroups", []) or []:
            if _field(row, "kind") == "village" and _field(row, "status") == "hold-for-review":
                villages.add(str(_field(row, "label")))
        for pid in _field(report, "held_ids", []) or []:
            held.add(int(pid))
    return {"drift_flags": len(villages), "held_decisions": len(held)}


def stream_advisories(reports) -> dict:
    """Labels (never person ids) that showed an advisory, with the number of batches each appeared in.

    An advisory is a flagged row that did not become a hold: an age band at or above the moderate
    PSI line, a village in the moderate band, or a confidence-drop warning. Advisories never add
    anyone to the hold list.
    """
    seen: dict[str, int] = {}
    for report in reports or []:
        for row in _field(report, "subgroups", []) or []:
            if _field(row, "flagged", False) and _field(row, "status") != "hold-for-review":
                label = str(_field(row, "label"))
                seen[label] = seen.get(label, 0) + 1
    return dict(sorted(seen.items()))


def score_batch(ss, batch: pd.DataFrame, base: Baseline) -> BatchReport:
    """Score one batch against a rolling 2-batch window kept on ``base``."""
    _fill_confidence(ss, base)
    frame = batch.reset_index(drop=True).copy()
    base.window.append(frame)
    if len(base.window) > _WINDOW:
        base.window = base.window[-_WINDOW:]
    window = pd.concat(base.window, ignore_index=True) if base.window else frame
    pred = predict(ss, window)
    conf = np.asarray(pred["confidence"], dtype=float)
    ent = np.asarray(pred["entropy"], dtype=float)
    rows = []
    psi_map: dict[str, float | None] = {}
    for label, kind, mask in _group_table(window):
        n = int(mask.sum())
        psi_map[label] = None
        if n < base.min_n:
            rows.append({
                "label": label, "kind": kind, "n": n,
                "mean_confidence": None, "mean_entropy": None, "max_psi": None,
                "flagged": False, "reason": "insufficient data", "status": "insufficient data",
            })
            continue
        feature_psi = []
        spec = base.specs.get(label, {})
        sub = window.loc[mask]
        for col, feature in spec.items():
            if col not in sub.columns:
                continue
            feature_psi.append(_psi_against(feature, sub[col].to_numpy()))
        max_psi = float(max(feature_psi)) if feature_psi else 0.0
        psi_map[label] = max_psi
        mean_c = float(conf[mask].mean())
        mean_e = float(ent[mask].mean())
        status = "ok"
        reason = "within baseline"
        flagged = False
        if kind == "village" and max_psi >= base.psi_major:
            status = "hold-for-review"
            reason = "village PSI at or above the major line"
            flagged = True
        elif max_psi >= base.psi_moderate:
            status = "warning"
            reason = "PSI in the moderate band" if kind == "village" else "age-band advisory"
            flagged = True
        if (base.confidence_mean is not None
                and mean_c < base.confidence_mean - base.confidence_drop):
            flagged = True
            if status == "ok":
                status = "warning"
                reason = "confidence drop"
            elif "confidence" not in reason:
                reason = reason + "; confidence drop"
        rows.append({
            "label": label, "kind": kind, "n": n,
            "mean_confidence": mean_c, "mean_entropy": mean_e, "max_psi": max_psi,
            "flagged": flagged, "reason": reason, "status": status,
        })
    held_villages = []
    for row in rows:
        if row["kind"] == "village" and row["status"] == "hold-for-review":
            code = int(row["label"].split("_")[1])
            held_villages.append(code)
    if held_villages and "person_id" in window.columns and "village_code" in window.columns:
        mask = window["village_code"].isin(held_villages)
        held_ids = [int(v) for v in window.loc[mask, "person_id"].tolist()]
    else:
        held_ids = []
    if any(row["status"] == "hold-for-review" for row in rows):
        outcome = "hold-for-review"
    elif any(row["flagged"] for row in rows):
        outcome = "warn"
    else:
        outcome = "proceed"
    flags = [row["label"] for row in rows if row["flagged"]]
    return BatchReport(
        subgroups=rows,
        held_ids=held_ids,
        psi=psi_map,
        entropy=float(ent.mean()) if len(ent) else None,
        outcome=outcome,
        flags=flags,
    )
