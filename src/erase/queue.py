"""Verifiable erasure: retrain only the affected shard(s), swap atomically,
audit before/after, log, and return a signed receipt (FR-4, FR-15).

Public API
----------
``erase(person_ids, request_id, ss, train_df)``   spec 8.3 signature (+ optional kwargs)
``EraseQueue(ss, train_df).submit(...)/.flush()`` FR-15 coalescing queue
``append_log`` / ``verify_log``                   re-exported from src.report.receipt
``load_train_df(data_dir)``                       train_current.csv if present, else train.csv

Order of effects inside one batch (chosen so a crash is always retryable):
  1. fail fast: key present, request ids valid, eval data readable
  2. audit BEFORE (old models) + subgroup AUROC BEFORE
  3. retrain ONLY affected shards in memory via ``shards.train_shard``
  4. time a real full retrain in a scratch dir (never swapped in)
  5. atomic swap: temp files -> os.replace -> manifest (temp + replace) -> memory
  6. persist data/train_current.csv (atomic; train.csv is never touched)
  7. audit AFTER (new models), test AUROC after, subgroup AUROC after
  8. append log + build + sign receipt (circular-hash fix lives in receipt.py)
A crash before step 6 leaves train_current.csv unchanged, so retrying the same
request still finds the rows and redoes the erase; nothing half-written is ever
served because model files only change through ``os.replace``.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import operator
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.runtime.resources import configure_threads, estimate_forest_mb, resolve_workers

configure_threads()

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.contracts import DATA_DIR, FEATURES, THRESHOLDS, ShardSet
from src.model import shards
from src.report import receipt as R
from src.report.receipt import append_log, verify_log  # noqa: F401  (spec 8.3 location)

__all__ = ["erase", "EraseQueue", "append_log", "verify_log", "load_train_df"]

# Captured at import so the scratch full retrain is NOT counted by tests that
# monkeypatch/count ``shards.train_shard`` (the real retrain goes through the
# module attribute, per requirement 4).
_REAL_TRAIN_SHARD = shards.train_shard

_ERASE_LOCK = threading.RLock()  # one swap/manifest write at a time, any caller


def _thr(name: str, default: float) -> float:
    """THRESHOLDS value; spec default only if the contract dict names it differently."""
    return THRESHOLDS.get(name, default)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def load_train_df(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Current training set: train_current.csv if an erase has happened, else train.csv."""
    d = Path(data_dir)
    cur = d / "train_current.csv"
    return pd.read_csv(cur if cur.exists() else d / "train.csv")


def _as_ids(person_ids: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in person_ids:
        if isinstance(x, (bool, np.bool_)):
            raise ValueError("person ids must be integers")
        try:
            i = operator.index(x)
        except TypeError as exc:
            raise ValueError(f"person ids must be integers, got {type(x).__name__}") from exc
        if i not in seen:
            seen.add(i)
            out.append(int(i))
    return out


def _sorted_shard_df(df: pd.DataFrame, shard_col: pd.Series, k: int) -> pd.DataFrame:
    return df[shard_col == k].sort_values("person_id", kind="mergesort").reset_index(drop=True)


def _shard_column(df: pd.DataFrame, n_shards: int) -> pd.Series:
    return df["person_id"].map(lambda p: shards.shard_of(int(p), n_shards))


def _label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:+\-]+", "_", s)[:40] or "x"


def _clean_metrics(obj: Any, depth: int = 0) -> Any:
    """Keep only numbers/None/bools (and short lists/dicts of them) so a surprise
    string or oversized list from another module can never break or pollute the receipt."""
    if depth > 5:
        return None
    if obj is None or isinstance(obj, (bool, np.bool_)):
        return None if obj is None else bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        if len(obj) > 64:
            return None
        return [_clean_metrics(v, depth + 1) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if not re.fullmatch(r"[a-z0-9_]{1,48}", ks) or ks.lower() in R.PATIENT_FIELD_DENYLIST:
                continue
            if isinstance(v, str):
                continue
            out[ks] = _clean_metrics(v, depth + 1)
        return out
    return None


def _clean_audit(a: Any) -> dict | None:
    if not isinstance(a, dict):
        return None
    out = {}
    for view in ("ensemble", "shard"):
        v = a.get(view)
        out[view] = _clean_metrics(v) if isinstance(v, dict) else None
    return out


def _view_auc(a: Any) -> float:
    try:
        v = a["shard"]["attack_auc"]
        return float(v) if v is not None and math.isfinite(float(v)) else float("-inf")
    except Exception:
        return float("-inf")


def _flatten_auroc(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        lab = _label(f"{prefix}{k}")
        if isinstance(v, dict):
            out.update(_flatten_auroc(v, lab + "_"))
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out[lab] = f
    return out


def _subgroup_auroc(ss: ShardSet, test_df: pd.DataFrame):
    """The Guard's subgroup_auroc, or (None, reason) if the Guard is unavailable/failed."""
    try:
        monitor = importlib.import_module("src.guard.monitor")
        fn = monitor.subgroup_auroc
    except (ImportError, AttributeError):
        return None, "Guard not available"
    try:
        return _flatten_auroc(fn(ss, test_df)), None
    except Exception as exc:  # an erasure that already happened must still get a receipt
        return None, f"Guard subgroup check failed ({type(exc).__name__})"


def _guard_counts(summary: Any) -> dict[str, int]:
    """Validate the optional Guard stream summary: counts only, never ids or features.

    Accepts ``None`` or a dict whose keys are within {"drift_flags", "held_decisions"} and whose
    values are non-negative integers. Anything else raises ValueError before an erase starts.
    """
    if summary is None:
        return {}
    if not isinstance(summary, dict) or not set(summary) <= {"drift_flags", "held_decisions"}:
        raise ValueError("guard_summary must be a dict with keys drift_flags and/or held_decisions")
    out: dict[str, int] = {}
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError(f"guard_summary[{key!r}] must be a non-negative count")
        out[key] = int(value)
    return out


def _guard_block(before, after, reason_b, reason_a, test_drop: float,
                 counts: dict[str, int] | None = None) -> tuple[dict, str | None]:
    counts = counts or {}
    null = {"worst_subgroup": None, "max_subgroup_auroc_drop": None,
            "drift_flags": counts.get("drift_flags"),
            "held_decisions": counts.get("held_decisions")}
    if before is None or after is None:
        return null, reason_b or reason_a
    try:
        from src.guard.monitor import compare_subgroup_auroc, flatten_subgroup_auroc
    except ImportError:
        return null, reason_b or reason_a or "Guard not available"
    flat_b = flatten_subgroup_auroc(before)
    flat_a = flatten_subgroup_auroc(after)
    common = sorted(set(flat_b) & set(flat_a))
    if not common:
        return null, "Guard returned no comparable subgroups"
    drops = {g: flat_b[g] - flat_a[g] for g in common}
    worst = max(common, key=lambda g: (drops[g], g))
    flagged = compare_subgroup_auroc(before, after, test_drop)
    block = {
        "worst_subgroup": worst,
        "max_subgroup_auroc_drop": float(drops[worst]),
        # Counts from a Guard stream run (monitor.stream_counts), or None when none was supplied.
        "drift_flags": counts.get("drift_flags"),
        "held_decisions": counts.get("held_decisions"),
        "flagged_subgroups": {g: round(float(d), 6) for g, d in flagged.items()},
    }
    return block, None


def _fsync_path(path: Path) -> None:
    """Flush a file that was just renamed into place.

    Opening read-only and calling os.fsync raises EBADF on Windows. A failed
    fsync must not undo the atomic os.replace that already landed the bytes.
    """
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _update_manifest(manifest: dict | None, new_hashes: dict[int, str]) -> dict:
    """Record new shard hashes without assuming the exact manifest layout."""
    m = copy.deepcopy(manifest) if manifest else {}
    touched = False
    if isinstance(m.get("shards"), dict):
        for k, h in new_hashes.items():
            ent = m["shards"].setdefault(str(k), {})
            if isinstance(ent, dict):
                ent["model_hash"] = h
                ent.pop("hash", None)
                touched = True
    for name in ("hashes", "model_hashes"):
        if isinstance(m.get(name), dict):
            for k, h in new_hashes.items():
                m[name][str(k)] = h
            touched = True
    if not touched:
        m.setdefault("hashes", {})
        for k, h in new_hashes.items():
            m["hashes"][str(k)] = h
    return m


def _atomic_swap(ss: ShardSet, new_models: dict[int, Any], new_hashes: dict[int, str]) -> None:
    """Serve either the old or the new model for each shard, never a partial file.

    1) write every new model to a temp file in the models dir and fsync it
    2) os.replace each temp file over shard_k.joblib (atomic rename)
    3) write the manifest the same way
    4) only then swap the in-memory models
    If step 1 fails nothing visible has changed and temp files are removed.
    """
    mdir = Path(ss.models_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    tmps: dict[int, Path] = {}
    try:
        for k, model in new_models.items():
            tmp = mdir / f".shard_{k}.joblib.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            tmps[k] = tmp
            joblib.dump(model, tmp)
            _fsync_path(tmp)
        manifest = _update_manifest(ss.manifest, new_hashes)
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    except BaseException:
        for tmp in tmps.values():
            if tmp.exists():
                tmp.unlink()
        raise
    for k, tmp in tmps.items():
        os.replace(tmp, mdir / f"shard_{k}.joblib")
    _atomic_write_bytes(mdir / "manifest.json", manifest_bytes)
    for k, model in new_models.items():
        ss.models[k] = model
    ss.manifest = manifest


def _fit_all_shards(new_train: pd.DataFrame, shard_col: pd.Series, n_shards: int,
                    n_workers: int) -> tuple[float, list]:
    """Time a full retrain that is never swapped in. Uses ``_REAL_TRAIN_SHARD``."""
    pairs = [(_sorted_shard_df(new_train, shard_col, k), k) for k in range(n_shards)]
    t0 = time.perf_counter()
    if n_workers <= 1:
        built = [_REAL_TRAIN_SHARD(df, k) for df, k in pairs]
    else:
        built = Parallel(n_jobs=n_workers, backend="loky", max_nbytes="1M", mmap_mode="r")(
            delayed(_REAL_TRAIN_SHARD)(df, k) for df, k in pairs)
    return time.perf_counter() - t0, list(built)


def _scratch_full_retrain(new_train: pd.DataFrame, shard_col: pd.Series, n_shards: int) -> tuple[float, float]:
    """Time a parallel full retrain and a serial one. Neither is swapped in.

    The parallel figure is the headline ``full_retrain_seconds`` (SC-12): the
    retrain a clinic would actually run, on every core ``plan_workers`` allows.
    The serial figure is ``full_retrain_serial_seconds``.
    """
    rows = max(1, int(len(new_train) / max(1, n_shards)))
    n_par = resolve_workers(-1, n_shards, estimate_forest_mb(rows))
    # _fit_all_shards calls _REAL_TRAIN_SHARD, captured at import, so a test that
    # monkeypatches shards.train_shard does not count the scratch retrain.
    with tempfile.TemporaryDirectory(prefix="forgetme_scratch_") as scratch:
        parallel_s, built = _fit_all_shards(new_train, shard_col, n_shards, n_par)
        for k, model in enumerate(built):  # scratch dir only; not served
            joblib.dump(model, Path(scratch) / f"scratch_shard_{k}.joblib")
        serial_s, _serial = _fit_all_shards(new_train, shard_col, n_shards, 1)
    return float(parallel_s), float(serial_s)


def _load_eval_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    test_p, held_p = d / "test.csv", d / "heldout_canaries.csv"
    for p in (test_p, held_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} is required for the erase audit/AUROC (run src/data/generate.py)")
    return pd.read_csv(test_p), pd.read_csv(held_p)


def _test_auroc(ss: ShardSet, test_df: pd.DataFrame) -> float:
    proba = np.asarray(shards.predict(ss, test_df[FEATURES])["proba"], dtype=float)
    return float(roc_auc_score(test_df["needs_referral"], proba))


# --------------------------------------------------------------------------- #
# core: one batch of requests
# --------------------------------------------------------------------------- #
def _erase_batch(
    requests: list[tuple[str, Any]],
    ss: ShardSet,
    train_df: pd.DataFrame,
    *,
    data_dir: str = DATA_DIR,
    log_path: str = R.DEFAULT_LOG_PATH,
    key: bytes | None = None,
    measure_full_retrain: bool = True,
    guard_summary: dict | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Erase every request in ONE pass: one retrain per affected shard in total.

    Returns (receipts in request order, remaining training set).
    """
    # ---- 1. fail fast, before anything is touched --------------------------
    key = R.get_hmac_key() if key is None else key
    reqs: list[tuple[str, list[int]]] = []
    seen_rids: set[str] = set()
    for rid, ids in requests:
        R.validate_request_id(rid)
        if rid in seen_rids:
            raise ValueError(f"duplicate request_id in one batch: {rid}")
        seen_rids.add(rid)
        reqs.append((rid, _as_ids(ids)))
    if not reqs:
        return [], train_df
    if "person_id" not in train_df.columns:
        raise ValueError("train_df needs a person_id column")
    R._require_key(key)
    guard_counts = _guard_counts(guard_summary)

    with _ERASE_LOCK:
        test_df, held_df = _load_eval_data(data_dir)
        n_shards = int(ss.n_shards)
        probe_rows = int(_thr("probe_rows", 1000))
        drop_thr = float(_thr("subgroup_auroc_drop", 0.03))
        probe = test_df[FEATURES].head(probe_rows)
        notes: list[str] = []

        # ---- locate rows and shards ---------------------------------------
        all_ids = {i for _, ids in reqs for i in ids}
        mask = train_df["person_id"].isin(all_ids)
        removed = train_df[mask].copy()
        found = {int(p) for p in removed["person_id"]}
        new_train = train_df[~mask].reset_index(drop=True)
        removed["_shard"] = _shard_column(removed, n_shards) if len(removed) else pd.Series(dtype=int)
        pid_shard = {int(p): int(s) for p, s in zip(removed["person_id"], removed["_shard"])}
        affected = sorted(set(pid_shard.values()))

        # attribute each person (and their rows) to the first request that names them
        assigned: set[int] = set()
        per_req = []
        for rid, ids in reqs:
            mine = [i for i in ids if i in found and i not in assigned]
            assigned.update(mine)
            per_req.append({
                "rid": rid,
                "persons": mine,
                "not_found": len(ids) - len(mine),
                "shards": sorted({pid_shard[i] for i in mine}),
                "rows": int(removed["person_id"].isin(mine).sum()) if mine else 0,
            })
        if any(r["not_found"] for r in per_req):
            notes.append("Some ids matched no remaining records")

        ts = R.utc_now()

        # ---- nothing to erase: still issue honest, logged receipts --------
        if not affected:
            auroc = _test_auroc(ss, test_df)
            notes.insert(0, "No matching records found; no model changed")
            receipts = []
            for r in per_req:
                pre = R.build_receipt(
                    request_id=r["rid"], timestamp_utc=ts, persons_erased=0, shard_ids=[],
                    rows_removed=0, model_hash_before={}, model_hash_after={},
                    full_retrain_seconds=0.0, shard_retrain_seconds=0.0, speedup=0.0,
                    test_auroc_before=auroc, test_auroc_after=auroc,
                    audit_before=None, audit_after=None,
                    persons_not_found=r["not_found"], batch_size=len(reqs),
                    note="; ".join(notes)[:200],
                )
                receipts.append(R.finalize_receipt(pre, key, log_path))
            return receipts, new_train

        new_shard_col = _shard_column(new_train, n_shards)
        empty = [k for k in affected if int((new_shard_col == k).sum()) == 0]
        if empty:
            raise ValueError(f"erasure would leave shard {empty[0]} with no training rows")

        # ---- 2. BEFORE measurements (old models) --------------------------
        auroc_before = _test_auroc(ss, test_df)
        hash_before = {k: shards.fingerprint(ss.models[k], probe) for k in affected}
        sub_before, why_b = _subgroup_auroc(ss, test_df)

        members_by_shard: dict[int, pd.DataFrame] = {}
        if "canary_type" in removed.columns:
            memb = removed[removed["canary_type"] == "member"]
            for k in affected:
                m = memb[memb["_shard"] == k].drop(columns=["_shard"]).reset_index(drop=True)
                if len(m):
                    members_by_shard[k] = m
        audit_b: dict[int, dict] = {}
        if members_by_shard:
            mia = importlib.import_module("src.audit.mia")
            for k, m in members_by_shard.items():
                audit_b[k] = mia.run_audit(ss, m, held_df, shard_id=k, data_dir=data_dir)
        else:
            notes.append("No member canaries among erased persons; audit not applicable")

        # ---- 3. retrain ONLY the affected shards (in memory) --------------
        def _timed_shard(df_k: pd.DataFrame, k: int):
            t0 = time.perf_counter()
            model = shards.train_shard(df_k, k)  # module attribute: monkeypatchable
            return k, model, time.perf_counter() - t0

        pairs = [(k, _sorted_shard_df(new_train, new_shard_col, k)) for k in affected]
        n_pool = resolve_workers(-1, len(pairs), estimate_forest_mb(max(1, int(len(new_train) / n_shards))))
        t_wall0 = time.perf_counter()
        if len(pairs) <= 1 or n_pool <= 1:
            trained = [_timed_shard(df_k, k) for k, df_k in pairs]
        else:
            # Threading, not loky: the served retrain must call the live
            # shards.train_shard attribute in this process so a counter wrapped
            # around that attribute sees every shard. The heavy pools (train_all,
            # shadows, the scratch full retrain) stay on loky.
            trained = Parallel(n_jobs=n_pool, backend="threading")(
                delayed(_timed_shard)(df_k, k) for k, df_k in pairs)
        shard_seconds = time.perf_counter() - t_wall0
        new_models = {k: model for k, model, _dt in trained}
        t_shard = {k: dt for k, _model, dt in trained}
        hash_after = {k: shards.fingerprint(new_models[k], probe) for k in affected}

        # ---- 4. real full retrain in a scratch dir, never swapped in ------
        serial_seconds = None
        if measure_full_retrain:
            full_seconds, serial_seconds = _scratch_full_retrain(new_train, new_shard_col, n_shards)
        else:
            full_seconds = 0.0
            notes.append("Full retrain not measured")
        # speedup is against the parallel full retrain (SC-12), not a serial one.
        speedup = (full_seconds / shard_seconds) if (measure_full_retrain and shard_seconds > 0) else 0.0
        speedup_vs_serial = (
            (serial_seconds / shard_seconds) if (serial_seconds and shard_seconds > 0) else None
        )

        # ---- 5. atomic swap ------------------------------------------------
        _atomic_swap(ss, new_models, hash_after)

        # ---- 6. persist the remaining training set (never train.csv) -------
        out_csv = Path(data_dir) / "train_current.csv"
        if out_csv.name == "train.csv":  # defensive; name is fixed above
            raise RuntimeError("refusing to overwrite train.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(out_csv, new_train.to_csv(index=False).encode("utf-8"))

        # ---- 7. AFTER measurements (new models) ---------------------------
        auroc_after = _test_auroc(ss, test_df)
        sub_after, why_a = _subgroup_auroc(ss, test_df)
        audit_a: dict[int, dict] = {}
        if members_by_shard:
            for k, m in members_by_shard.items():
                audit_a[k] = mia.run_audit(ss, m, held_df, shard_id=k, data_dir=data_dir)
        guard, guard_note = _guard_block(sub_before, sub_after, why_b, why_a, drop_thr, guard_counts)
        if guard_note:
            notes.append(guard_note)

        # ---- 8. log + sign, one receipt per request -----------------------
        receipts = []
        for r in per_req:
            ks = r["shards"]
            audited = [k for k in ks if k in audit_b]
            top = max(audited, key=lambda k: (_view_auc(audit_b[k]), -k)) if audited else None
            detail = {
                str(k): {
                    "audit_before": _clean_audit(audit_b.get(k)),
                    "audit_after": _clean_audit(audit_a.get(k)),
                    "retrain_seconds": float(t_shard[k]),
                    "members_audited": int(len(members_by_shard[k])) if k in members_by_shard else 0,
                }
                for k in ks
            }
            req_notes = list(notes)
            if ks and not audited and "No member canaries among erased persons; audit not applicable" not in req_notes:
                req_notes.append("No member canaries among erased persons; audit not applicable")
            fields = dict(
                request_id=r["rid"], timestamp_utc=ts,
                persons_erased=len(r["persons"]), shard_ids=ks, rows_removed=r["rows"],
                model_hash_before={str(k): hash_before[k] for k in ks},
                model_hash_after={str(k): hash_after[k] for k in ks},
                full_retrain_seconds=float(full_seconds),
                shard_retrain_seconds=float(shard_seconds),
                speedup=float(speedup),
                test_auroc_before=auroc_before, test_auroc_after=auroc_after,
                audit_before=_clean_audit(audit_b[top]) if top is not None else None,
                audit_after=_clean_audit(audit_a[top]) if top is not None else None,
                guard=guard, shard_detail=detail if ks else None,
                persons_not_found=r["not_found"], batch_size=len(reqs),
            )
            if serial_seconds is not None:
                fields["full_retrain_serial_seconds"] = float(serial_seconds)
                if speedup_vs_serial is not None:
                    fields["speedup_vs_serial"] = float(speedup_vs_serial)
            if req_notes:
                fields["note"] = "; ".join(req_notes)[:200]
            pre = R.build_receipt(**fields)
            receipts.append(R.finalize_receipt(pre, key, log_path))
        return receipts, new_train


def erase(
    person_ids: list[int],
    request_id: str,
    ss: ShardSet,
    train_df: pd.DataFrame,
    *,
    data_dir: str = DATA_DIR,
    log_path: str = R.DEFAULT_LOG_PATH,
    key: bytes | None = None,
    measure_full_retrain: bool = True,
    guard_summary: dict | None = None,
) -> dict:
    """Erase ``person_ids`` and return the signed receipt (spec 8.3).

    ``guard_summary`` (optional) is ``{"drift_flags": int, "held_decisions": int}`` from
    ``src.guard.monitor.stream_counts``. It is copied onto ``receipt["guard"]`` as counts only.

    Only the shards that held these people are retrained - exactly once each,
    however many ids fall in the same shard. ``ss`` is updated in place and the
    remaining training set is saved to ``<data_dir>/train_current.csv``
    (``train.csv`` is never modified); reload it with ``load_train_df``.
    """
    receipts, _ = _erase_batch(
        [(request_id, person_ids)], ss, train_df, data_dir=data_dir, log_path=log_path,
        key=key, measure_full_retrain=measure_full_retrain, guard_summary=guard_summary,
    )
    return receipts[0]


# --------------------------------------------------------------------------- #
# FR-15: coalescing queue
# --------------------------------------------------------------------------- #
class EraseQueue:
    """Thread-safe queue that coalesces erase requests.

    ``submit`` is cheap and never retrains. ``flush`` groups everything pending,
    retrains each affected shard ONCE (cost scales with shards touched, not with
    the number of requests) and returns one signed receipt per request, in
    submission order. Each receipt carries the shared batch timings, test AUROC,
    guard block and the audits/hashes of the shards that request touched, plus
    ``batch_size``.

    The constructor holds the live ShardSet and training set (after each flush the
    queue's ``train_df`` is the remaining data).
    """

    def __init__(
        self,
        ss: ShardSet,
        train_df: pd.DataFrame | None = None,
        *,
        data_dir: str = DATA_DIR,
        log_path: str = R.DEFAULT_LOG_PATH,
        key: bytes | None = None,
        measure_full_retrain: bool = True,
        guard_summary: dict | None = None,
    ) -> None:
        self.ss = ss
        self.train_df = load_train_df(data_dir) if train_df is None else train_df
        self.data_dir = data_dir
        self.log_path = log_path
        self.key = key
        self.measure_full_retrain = measure_full_retrain
        self.guard_summary = _guard_counts(guard_summary) or None
        self._pending: list[tuple[str, list[int]]] = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()

    def submit(self, person_ids: list[int], request_id: str) -> None:
        R.validate_request_id(request_id)
        ids = _as_ids(person_ids)
        with self._lock:
            if any(rid == request_id for rid, _ in self._pending):
                raise ValueError(f"request_id already pending: {request_id}")
            self._pending.append((request_id, ids))

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def flush(self, guard_summary: dict | None = None) -> list[dict]:
        """Retrain once per affected shard. ``guard_summary`` overrides the constructor's counts."""
        summary = self.guard_summary if guard_summary is None else guard_summary
        with self._flush_lock:
            with self._lock:
                batch, self._pending = self._pending, []
            if not batch:
                return []
            try:
                receipts, new_train = _erase_batch(
                    batch, self.ss, self.train_df, data_dir=self.data_dir,
                    log_path=self.log_path, key=self.key,
                    measure_full_retrain=self.measure_full_retrain, guard_summary=summary,
                )
            except BaseException:
                with self._lock:  # nothing was lost: requests go back to the front
                    self._pending = batch + self._pending
                raise
            self.train_df = new_train
            return receipts
