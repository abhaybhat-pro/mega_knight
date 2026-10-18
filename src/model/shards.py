"""8-shard RandomForest layer (FR-2, FR-3).

Determinism: sha256 sharding, rows sorted by person_id before fitting, shard k
uses random_state SEED+k, n_jobs=1 inside forests, parallelism only across shards.
Models are never committed (models/ is git-ignored); they are rebuilt on the VM.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

from src.runtime.resources import configure_threads, estimate_forest_mb, resolve_workers

configure_threads()

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import xlogy
from sklearn.ensemble import RandomForestClassifier

from src.contracts import (
    DATA_DIR, FEATURES, ID_COL, LABEL_COL, MANIFEST_NAME, MODELS_DIR, N_SHARDS,
    PERSON_MAP_NAME, RF_PARAMS, SEED, TEST_CSV, THRESHOLDS, ShardSet,
)


# ------------------------------------------------------------------ sharding
def shard_of(person_id: int, n_shards: int = N_SHARDS) -> int:
    """sha256-based shard id. Never uses Python's built-in hash()."""
    digest = hashlib.sha256(str(int(person_id)).encode("utf-8")).hexdigest()
    return int(digest, 16) % n_shards


def assign_shards(person_ids, n_shards: int = N_SHARDS) -> np.ndarray:
    return np.array([shard_of(p, n_shards) for p in person_ids], dtype=np.int64)


def person_shard_map(train_df: pd.DataFrame, n_shards: int = N_SHARDS) -> dict:
    return {int(p): shard_of(p, n_shards) for p in train_df[ID_COL]}


# ------------------------------------------------------------------ training
def train_shard(df_shard: pd.DataFrame, shard_id: int,
                seed: int = SEED) -> RandomForestClassifier:
    """Fit one shard's forest. Rows are sorted by person_id first (stable)."""
    if len(df_shard) == 0:
        raise ValueError(f"shard {shard_id} has no rows")
    df = df_shard.sort_values(ID_COL, kind="stable")
    rf = RandomForestClassifier(random_state=seed + shard_id, **RF_PARAMS)
    rf.fit(df[FEATURES], df[LABEL_COL].to_numpy())
    return rf


def fit_models(train_df: pd.DataFrame, n_shards: int = N_SHARDS,
               n_workers: int = -1, seed: int = SEED) -> dict:
    """Train all shards in a loky process pool, one task per shard. No disk I/O.

    Returns {shard_id: model}. Used by train_all and by shards_curve.
    """
    sid = assign_shards(train_df[ID_COL], n_shards)
    parts = [train_df.loc[sid == k] for k in range(n_shards)]
    empty = [k for k, p in enumerate(parts) if len(p) == 0]
    if empty:
        raise ValueError(f"empty shards {empty}; too few rows for n_shards={n_shards}")
    rows = max((len(p) for p in parts), default=1)
    n_jobs = resolve_workers(n_workers, n_shards, estimate_forest_mb(rows))
    # One flat pool. n_workers=-1 is plan_workers, not joblib's "all cores".
    models = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M", mmap_mode="r")(
        delayed(train_shard)(p, k, seed) for k, p in enumerate(parts))
    return {k: m for k, m in enumerate(models)}


# --------------------------------------------------------------- persistence
def _atomic_write_json(obj, path: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, sort_keys=True, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_shard(model, shard_id: int, models_dir: str = MODELS_DIR) -> str:
    """Atomic save: temp file in the same directory, then os.replace."""
    os.makedirs(models_dir, exist_ok=True)
    final = os.path.join(models_dir, f"shard_{shard_id}.joblib")
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(model, tmp)
        os.replace(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return final


def load_probe(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """The fixed probe set: first 1000 rows of data/test.csv."""
    path = os.path.join(data_dir, TEST_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run `python -m src.data.generate`")
    return pd.read_csv(path).head(THRESHOLDS["probe_rows"])


def build_manifest(models: dict, row_counts: dict, probe_X: pd.DataFrame,
                   n_shards: int, seed: int = SEED) -> dict:
    """Counts and hashes only, no patient content. Keys are strings (JSON)."""
    shards = {
        str(k): {"file": f"shard_{k}.joblib", "rows": int(row_counts[k]),
                 "model_hash": fingerprint(models[k], probe_X)}
        for k in sorted(models)
    }
    ens = ShardSet(models=models, n_shards=n_shards, models_dir="")
    return {
        "schema_version": 1, "n_shards": n_shards, "seed": seed,
        "total_rows": int(sum(row_counts.values())),
        "ensemble_hash": fingerprint(ens, probe_X),
        "person_map_file": PERSON_MAP_NAME,
        "shards": shards,
    }


def train_all(train_df: pd.DataFrame, n_shards: int = N_SHARDS, n_workers: int = -1,
              models_dir: str = MODELS_DIR, probe_X: pd.DataFrame | None = None,
              seed: int = SEED) -> ShardSet:
    """Train every shard, save models/shard_k.joblib, manifest.json and the
    person-to-shard map. ``models_dir``, ``probe_X`` and ``seed`` are optional extras."""
    models = fit_models(train_df, n_shards, n_workers, seed)
    if probe_X is None:
        probe_X = load_probe()
    os.makedirs(models_dir, exist_ok=True)
    keep = {f"shard_{k}.joblib" for k in models}
    for name in os.listdir(models_dir):  # stale shard files from another n_shards
        if name.startswith("shard_") and name.endswith(".joblib") and name not in keep:
            os.remove(os.path.join(models_dir, name))
    for k, m in models.items():
        save_shard(m, k, models_dir)
    sid = assign_shards(train_df[ID_COL], n_shards)
    counts = {k: int((sid == k).sum()) for k in range(n_shards)}
    manifest = build_manifest(models, counts, probe_X, n_shards, seed)
    _atomic_write_json(manifest, os.path.join(models_dir, MANIFEST_NAME))
    pmap = {str(p): int(s) for p, s in zip(train_df[ID_COL], sid)}
    _atomic_write_json(pmap, os.path.join(models_dir, PERSON_MAP_NAME))
    return ShardSet(models=models, n_shards=n_shards, models_dir=models_dir,
                    manifest=manifest)


def load(models_dir: str = MODELS_DIR) -> ShardSet:  # noqa: D401 — default is the repo models dir
    path = os.path.join(models_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; train first (train_all)")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    n = int(manifest["n_shards"])
    models = {k: joblib.load(os.path.join(models_dir, f"shard_{k}.joblib"))
              for k in range(n)}
    return ShardSet(models=models, n_shards=n, models_dir=models_dir, manifest=manifest)


# ---------------------------------------------------------------- prediction
def _p1(model, X: pd.DataFrame) -> np.ndarray:
    """P(class 1) for one forest, robust to a single-class shard."""
    proba = model.predict_proba(X[FEATURES])
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)].astype(np.float64)
    return np.zeros(len(X), dtype=np.float64)


def predict(ss: ShardSet, X: pd.DataFrame) -> dict:
    """Ensemble prediction.

    Returns dict with:
      proba      ndarray (n,)  mean of per_shard
      label      ndarray (n,)  int, 1 if proba >= referral_threshold
      confidence ndarray (n,)  max(p, 1-p)
      entropy    ndarray (n,)  binary Shannon entropy in bits
      per_shard  dict[int, ndarray (n,)]  P(class 1) from each shard, keyed by shard id
    """
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    per_shard = {int(k): _p1(m, X) for k, m in sorted(ss.models.items())}
    p = np.mean(np.vstack(list(per_shard.values())), axis=0)
    p = np.clip(p, 0.0, 1.0)
    ent = -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)) / np.log(2.0)
    return {
        "proba": p,
        "label": (p >= THRESHOLDS["referral_threshold"]).astype(np.int64),
        "confidence": np.maximum(p, 1.0 - p),
        "entropy": ent,
        "per_shard": per_shard,
    }


def per_shard_matrix(pred: dict) -> np.ndarray:
    """per_shard dict as an (n_shards, n) array ordered by shard id."""
    return np.vstack([pred["per_shard"][k] for k in sorted(pred["per_shard"])])


def fingerprint(model, probe_X: pd.DataFrame) -> str:
    """Model hash: sha256 of predict_proba on the probe set, rounded to 8 decimals.

    ``model`` may be one RandomForestClassifier or a ShardSet (ensemble proba).
    Bytes are hashed as little-endian float64, so the hash does not depend on the
    host's byte order (s390x is big-endian).
    """
    if isinstance(model, ShardSet):
        p = predict(model, probe_X)["proba"]
    else:
        p = _p1(model, probe_X)
    p = np.round(np.asarray(p, dtype=np.float64), THRESHOLDS["probe_decimals"]) + 0.0
    return hashlib.sha256(p.astype("<f8").tobytes()).hexdigest()
