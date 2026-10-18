"""Synthetic clinic data (FR-1). Synthetic data only.

Writes data/train.csv, data/test.csv, data/heldout_canaries.csv, data/stream_base.csv.
Fully seeded: two runs give byte-identical CSVs. Floats are rounded before
writing so last-ulp differences between architectures cannot change the bytes.
"""
from __future__ import annotations

import os
from typing import Iterator

from src.runtime.resources import configure_threads

configure_threads()

import numpy as np
import pandas as pd

from src.contracts import (
    CANARY_COL, COLUMNS, DATA_DIR, FEATURES, HELDOUT_CSV, ID_BASE, ID_COL,
    LABEL_COL, N_CANARY, N_PATIENTS, N_STREAM_BASE, N_TEST, N_VILLAGES,
    RARE_VILLAGES, SEED, STREAM_CSV, TEST_CSV, THRESHOLDS, TRAIN_CSV,
)

LOGIT_INTERCEPT = -1.9   # tuned for roughly 25-30% referral prevalence
LABEL_NOISE_SD = 0.6
# village 3 is ~25% of ordinary rows; the two rare villages are ~3% each
_VILLAGE_P = np.array(
    [0.138, 0.138, 0.138, 0.25, 0.138, 0.138, 0.03, 0.03], dtype=float)
assert len(_VILLAGE_P) == N_VILLAGES and abs(_VILLAGE_P.sum() - 1.0) < 1e-12


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Ordinary patients' features (no id, no label). All in normal ranges."""
    age = np.clip(np.rint(rng.normal(52, 16, n)), 18, 90).astype(np.int64)
    sex = rng.integers(0, 2, n, dtype=np.int64)
    bmi = np.clip(rng.normal(27, 4.5, n), 16, 39.5)
    sbp = np.clip(rng.normal(118 + 0.35 * (age - 50) + 0.5 * (bmi - 27), 13), 85, 200)
    dbp = np.clip(rng.normal(76 + 0.15 * (age - 50) + 0.3 * (bmi - 27), 8), 50, 120)
    hba1c = np.clip(rng.normal(5.6 + 0.03 * (bmi - 27) + 0.008 * (age - 50), 0.8),
                    4.0, 10.5)
    smoker = (rng.random(n) < 0.22).astype(np.int64)
    hx = np.maximum(hba1c - 5.6, 0.0)
    fatigue = (rng.random(n) < np.clip(0.15 + 0.03 * hx, 0, 0.9)).astype(np.int64)
    vision = (rng.random(n) < np.clip(0.08 + 0.04 * hx, 0, 0.9)).astype(np.int64)
    thirst = (rng.random(n) < np.clip(0.08 + 0.10 * hx, 0, 0.9)).astype(np.int64)
    chest = (rng.random(n) < np.clip(0.03 + 0.001 * (age - 18), 0, 0.9)).astype(np.int64)
    village = rng.choice(N_VILLAGES, size=n, p=_VILLAGE_P).astype(np.int64)
    prior = np.clip(rng.poisson(1.5, n), 0, 15).astype(np.int64)
    # keep ordinary rows clear of the canary "rare village + 3 symptoms" pattern
    clash = np.isin(village, RARE_VILLAGES) & (chest == 1) & (vision == 1) & (thirst == 1)
    thirst = np.where(clash, 0, thirst)
    return pd.DataFrame({
        "age": age, "sex": sex,
        "bmi": np.round(bmi, 1), "sbp": np.round(sbp, 1), "dbp": np.round(dbp, 1),
        "hba1c": np.round(hba1c, 2), "smoker": smoker,
        "sym_fatigue": fatigue, "sym_vision": vision, "sym_thirst": thirst,
        "sym_chest_pain": chest, "village_code": village, "prior_visits": prior,
    })


def _label(rng: np.random.Generator, f: pd.DataFrame) -> np.ndarray:
    z = (LOGIT_INTERCEPT
         + 0.03 * (f["age"] - 50) + 0.07 * (f["bmi"] - 27) + 0.015 * (f["sbp"] - 120)
         + 0.9 * (f["hba1c"] - 5.6) + 0.45 * f["smoker"]
         + 0.5 * f["sym_fatigue"] + 0.4 * f["sym_vision"] + 0.6 * f["sym_thirst"]
         + 1.0 * f["sym_chest_pain"] + 0.12 * f["prior_visits"]
         + np.where(f["village_code"] == 6, 0.6, 0.0)
         + np.where(f["village_code"] == 7, -0.5, 0.0))
    z = np.asarray(z, dtype=float) + rng.normal(0, LABEL_NOISE_SD, len(f))
    return (rng.random(len(f)) < _sigmoid(z)).astype(np.int64)


def _ordinary(rng, n: int, id_start: int) -> pd.DataFrame:
    f = _features(rng, n)
    f.insert(0, ID_COL, np.arange(id_start, id_start + n, dtype=np.int64))
    f[LABEL_COL] = _label(rng, f)
    f[CANARY_COL] = "none"
    return f[COLUMNS]


def _canaries(rng, n: int, id_start: int, kind: str) -> pd.DataFrame:
    """Out-of-range rows with random Bernoulli(0.5) labels (unlearnable except by
    memorisation). Three patterns, assigned in a seeded random order:
      0: age < 25 and bmi > 40      1: hba1c > 11
      2: rare village with chest pain + vision + thirst symptoms
    """
    f = _features(rng, n)
    pattern = rng.permutation(np.arange(n) % 3)
    p0, p1, p2 = pattern == 0, pattern == 1, pattern == 2
    n0, n1, n2 = int(p0.sum()), int(p1.sum()), int(p2.sum())
    f.loc[p0, "age"] = rng.integers(18, 25, n0, dtype=np.int64)
    f.loc[p0, "bmi"] = np.round(rng.uniform(40.5, 55.0, n0), 1)
    f.loc[p1, "hba1c"] = np.round(rng.uniform(11.2, 15.0, n1), 2)
    f.loc[p2, "village_code"] = rng.choice(RARE_VILLAGES, n2).astype(np.int64)
    for col in ("sym_chest_pain", "sym_vision", "sym_thirst"):
        f.loc[p2, col] = 1
    # make every canary feature vector distinct and unusual-ish (not label-linked)
    f["sbp"] = np.round(rng.uniform(85, 200, n), 1)
    f["dbp"] = np.round(rng.uniform(50, 120, n), 1)
    f.insert(0, ID_COL, np.arange(id_start, id_start + n, dtype=np.int64))
    f[LABEL_COL] = rng.integers(0, 2, n, dtype=np.int64)
    f[CANARY_COL] = kind
    return f[COLUMNS]


def is_out_of_range(df: pd.DataFrame) -> pd.Series:
    """True where a row matches a canary pattern (used by tests and the audit)."""
    a = (df["bmi"] > 40) & (df["age"] < 25)
    b = df["hba1c"] > 11
    c = (df["village_code"].isin(RARE_VILLAGES) & (df["sym_chest_pain"] == 1)
         & (df["sym_vision"] == 1) & (df["sym_thirst"] == 1))
    return a | b | c


def _write(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def generate(seed: int = SEED, out_dir: str = DATA_DIR) -> None:
    """Write the four CSVs. ``out_dir`` is an optional extra parameter."""
    os.makedirs(out_dir, exist_ok=True)
    r_train, r_mem, r_held, r_test, r_stream = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5))
    ordinary = _ordinary(r_train, N_PATIENTS - N_CANARY, ID_BASE["train_ordinary"])
    member = _canaries(r_mem, N_CANARY, ID_BASE["member"], "member")
    held = _canaries(r_held, N_CANARY, ID_BASE["heldout"], "heldout")
    train = (pd.concat([ordinary, member], ignore_index=True)
             .sort_values(ID_COL, kind="stable").reset_index(drop=True))
    _write(train, os.path.join(out_dir, TRAIN_CSV))
    _write(held, os.path.join(out_dir, HELDOUT_CSV))
    _write(_ordinary(r_test, N_TEST, ID_BASE["test"]), os.path.join(out_dir, TEST_CSV))
    _write(_ordinary(r_stream, N_STREAM_BASE, ID_BASE["stream"]),
           os.path.join(out_dir, STREAM_CSV))


def read(name: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read one of the four CSVs by file name, e.g. read(TRAIN_CSV)."""
    return pd.read_csv(os.path.join(data_dir, name))


def drift_stream(base: pd.DataFrame, village: int = THRESHOLDS["drift_village"],
                 feature: str = THRESHOLDS["drift_feature"],
                 shift: float = THRESHOLDS["drift_shift"],
                 start_batch: int = THRESHOLDS["drift_start_batch"],
                 batch_size: int = THRESHOLDS["batch_size"],
                 n_batches: int = THRESHOLDS["n_batches"]) -> Iterator[pd.DataFrame]:
    """Yield ``n_batches`` batches of ``batch_size`` rows taken in order from base.

    From ``start_batch`` (0-based) on, ``shift`` is added to ``feature`` for rows in
    ``village`` only. shift == 0, or start_batch >= n_batches, gives the plain
    no-drift stream. If base is shorter than n_batches*batch_size it wraps around.
    """
    n = len(base)
    if n == 0:
        raise ValueError("base is empty")
    for i in range(n_batches):
        idx = (np.arange(i * batch_size, (i + 1) * batch_size)) % n
        batch = base.iloc[idx].reset_index(drop=True).copy()
        if shift != 0 and i >= start_batch:
            mask = batch["village_code"] == village
            batch.loc[mask, feature] = batch.loc[mask, feature] + shift
        yield batch


if __name__ == "__main__":
    generate()
    print(f"wrote 4 CSVs to {DATA_DIR}/ (seed={SEED}, features={len(FEATURES)})")
