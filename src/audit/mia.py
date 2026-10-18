"""Membership-inference audit Evidence engine for ForgetMe.

Public contract (spec 8.3):  run_audit(ss, members, nonmembers, shard_id=None) -> dict
Extra OPTIONAL parameter:    data_dir="data"  (needed once, to build the shadow cache)

Design notes
------------
* Score = logit of the model's probability of the record's TRUE label, clipped to [0.005, 0.995].
* Shadows: 16 ensembles. Shadows 2i and 2i+1 use complementary random halves of the 800
  canaries, so every canary is IN exactly 8 shadows and OUT of exactly 8. Only SCORES are cached
  (models/shadow/*.npz), never features or model binaries.
* Per canary: Gaussian in/out fit. Variance = per-canary pooled variance shrunk toward the
  median pooled variance across canaries (k0 pseudo-observations), floored at SIGMA_MIN**2.
  Attack score = log N(x; mu_in, s2) - log N(x; mu_out, s2).
* attack_auc / auc_ci95 are the SHADOW-CALIBRATED values; *_raw keys carry the uncalibrated ones.
* The cache key is the content hash of train.csv and heldout_canaries.csv plus the seed.
  An erase writes train_current.csv and does not rewrite train.csv, so a shard retrain
  does not change the key. If the cache is missing and a requested canary is not in the
  pool, we raise instead of guessing.
* Bootstrap CIs resample the audited scores only (shadow-estimation uncertainty is NOT included).
"""
from __future__ import annotations

import hashlib
import os
import tempfile

from src.runtime.resources import (
    assert_artifact_budget,
    configure_threads,
    estimate_forest_mb,
    resolve_workers,
)

configure_threads()

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.stats import rankdata

from src.contracts import DATA_DIR, FEATURES, MODELS_DIR, SEED, SHADOW_DIR, THRESHOLDS
from src.model.shards import shard_of, train_shard

N_SHADOWS = 16
CLIP_LO, CLIP_HI = 0.005, 0.995
SIGMA_MIN = 0.10      # logit units
SHRINK_K0 = 4.0       # pseudo-dof pulling per-canary variance toward the global median
BOOT_N = int(THRESHOLDS["n_bootstrap"])
FPR = 0.01

_CACHE: dict = {}


# ----------------------------------------------------------------------------- scoring
def logit_true(p_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.where(np.asarray(y) == 1, p_pos, 1.0 - p_pos)
    p = np.clip(p, CLIP_LO, CLIP_HI)
    return np.log(p / (1.0 - p))


def proba_pos(model, X: pd.DataFrame) -> np.ndarray:
    P = model.predict_proba(X)
    classes = list(model.classes_)
    return P[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def target_scores(ss, df: pd.DataFrame, level: str, shard_id: int | None = None) -> np.ndarray:
    """Raw (uncalibrated) logit-true scores of df under the shard model or the deployed ensemble."""
    X, y = df[FEATURES], df["needs_referral"].values
    if level == "shard":
        p = proba_pos(ss.models[shard_id], X)
    else:
        p = np.mean([proba_pos(ss.models[k], X) for k in sorted(ss.models)], axis=0)
    return logit_true(p, y)


# ----------------------------------------------------------------------------- shadows
def shadow_cache_key(data_dir: str, n_shards: int, seed: int = SEED) -> str:
    """Hash of the shadow training files plus the seed.

    The deployed shard weights are not an input, so retraining a shard does not
    change the key. ``train.csv`` is never rewritten by an erase (the remaining
    rows go to ``train_current.csv``), so an erase does not change it either.
    """
    h = hashlib.sha256()
    for name in ("train.csv", "heldout_canaries.csv"):
        with open(os.path.join(data_dir, name), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    h.update(f"|{int(seed)}|{int(n_shards)}|{N_SHADOWS}|{sklearn.__version__}".encode())
    return h.hexdigest()[:16]


def _shadow_key(heldout_ids, n_shards: int, data_dir: str | None = None) -> str:
    """Backward-compatible name. Prefer ``shadow_cache_key`` (content hash + seed)."""
    if data_dir is not None:
        return shadow_cache_key(data_dir, n_shards, SEED)
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in sorted(heldout_ids)).encode())
    h.update(f"|{SEED}|{n_shards}|{N_SHADOWS}|{sklearn.__version__}".encode())
    return h.hexdigest()[:16]


def _frame(ids, matrix, y, columns) -> pd.DataFrame:
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "person_id", np.asarray(ids, dtype=np.int64))
    df["needs_referral"] = np.asarray(y)
    return df


def _shadow_shard_scores(j, k, ord_ids, ord_x, ord_y, can_ids, can_x, can_y, in_row, n_shards):
    """Train one shadow shard and return canary probabilities. The forest is not returned."""
    take = np.asarray(in_row, dtype=bool)
    ids = np.concatenate([ord_ids, can_ids[take]])
    x = np.vstack([ord_x, can_x[take]])
    y = np.concatenate([ord_y, can_y[take]])
    df = _frame(ids, x, y, FEATURES)
    sh = np.array([shard_of(int(p), n_shards) for p in df["person_id"]])
    part = df.loc[sh == k]
    model = train_shard(part, k, seed=SEED + 1000 + 100 * j)
    can = _frame(can_ids, can_x, can_y, FEATURES)
    return int(j), int(k), proba_pos(model, can[FEATURES])


def _build_shadows(ss, data_dir: str, path: str, n_jobs: int = -1) -> dict:
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    held = pd.read_csv(os.path.join(data_dir, "heldout_canaries.csv"))
    ordinary = train[train["canary_type"] == "none"]
    canaries = (pd.concat([train[train["canary_type"] == "member"], held])
                .sort_values("person_id").reset_index(drop=True))
    c_count = len(canaries)
    rng = np.random.RandomState(SEED + 777)
    inmask = np.zeros((N_SHADOWS, c_count), dtype=bool)
    for i in range(N_SHADOWS // 2):
        perm = rng.permutation(c_count)
        inmask[2 * i, perm[: c_count // 2]] = True
        inmask[2 * i + 1, perm[c_count // 2:]] = True
    ord_ids = ordinary["person_id"].to_numpy(dtype=np.int64)
    ord_x = ordinary[FEATURES].to_numpy(dtype=np.float64)
    ord_y = ordinary["needs_referral"].to_numpy()
    can_ids = canaries["person_id"].to_numpy(dtype=np.int64)
    can_x = canaries[FEATURES].to_numpy(dtype=np.float64)
    can_y = canaries["needs_referral"].to_numpy()
    # Largest ordinary shard first, so the long tasks start before the short tail.
    sizes = np.array([int(np.sum([shard_of(int(p), ss.n_shards) == k for p in ord_ids]))
                      for k in range(ss.n_shards)])
    order = list(np.argsort(-sizes))
    tasks = []
    for k in order:
        for j in range(N_SHADOWS):
            tasks.append(delayed(_shadow_shard_scores)(
                j, int(k), ord_ids, ord_x, ord_y, can_ids, can_x, can_y, inmask[j], ss.n_shards))
    n_pool = resolve_workers(n_jobs, len(tasks), estimate_forest_mb(max(1, int(sizes.max()) if len(sizes) else 1)))
    pool_kw = dict(n_jobs=n_pool, backend="loky", max_nbytes="1M", mmap_mode="r")
    try:
        iterator = Parallel(return_as="generator", **pool_kw)(tasks)
    except TypeError:
        iterator = Parallel(**pool_kw)(tasks)
    probs = np.empty((N_SHADOWS, ss.n_shards, c_count), dtype=np.float64)
    for j, k, proba in iterator:
        probs[j, k, :] = proba
    y = can_y
    ens_p = probs.mean(axis=1)
    cshard = np.array([shard_of(int(p), ss.n_shards) for p in can_ids])
    shd_p = probs[np.arange(N_SHADOWS)[:, None], cshard[None, :], np.arange(c_count)[None, :]]
    out = dict(
        ens=np.stack([logit_true(ens_p[j], y) for j in range(N_SHADOWS)]),
        shd=np.stack([logit_true(shd_p[j], y) for j in range(N_SHADOWS)]),
        inmask=inmask,
        ids=can_ids.astype(np.int64),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert_artifact_budget(c_count * N_SHADOWS * 16, [data_dir, os.path.dirname(path), MODELS_DIR, SHADOW_DIR])
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz")
    os.close(fd)
    np.savez(tmp, **out)
    os.replace(tmp, path)                       # atomic: never a half-written cache
    return out


def get_shadows(ss, data_dir: str = DATA_DIR, n_jobs: int = -1) -> dict:
    """Load the cached shadow scores, building them once if absent."""
    key = shadow_cache_key(data_dir, ss.n_shards, SEED)
    path = os.path.join(ss.models_dir, "shadow", f"shadow_{key}.npz")
    if path in _CACHE:
        return _CACHE[path]
    if os.path.exists(path):
        z = np.load(path)
        sh = {k: z[k] for k in z.files}
    else:
        sh = _build_shadows(ss, data_dir, path, n_jobs)
    sh["index"] = {int(p): i for i, p in enumerate(sh["ids"])}
    _CACHE[path] = sh
    return sh


def _calibration(S: np.ndarray, inmask: np.ndarray):
    """Per-canary in/out Gaussian parameters. S, inmask: (J, C)."""
    out = ~inmask
    n_in, n_out = inmask.sum(0), out.sum(0)
    mu_in = (S * inmask).sum(0) / n_in
    mu_out = (S * out).sum(0) / n_out
    ssq = (((S - mu_in) ** 2) * inmask).sum(0) + (((S - mu_out) ** 2) * out).sum(0)
    dof = np.maximum(n_in + n_out - 2, 1)
    var_c = ssq / dof
    var_g = np.median(var_c)
    var = (dof * var_c + SHRINK_K0 * var_g) / (dof + SHRINK_K0)
    return mu_in, mu_out, np.maximum(var, SIGMA_MIN ** 2)


def llr_scores(x: np.ndarray, idx: np.ndarray, S: np.ndarray, inmask: np.ndarray) -> np.ndarray:
    mu_in, mu_out, var = _calibration(S, inmask)
    mi, mo, v = mu_in[idx], mu_out[idx], var[idx]
    return ((x - mo) ** 2 - (x - mi) ** 2) / (2.0 * v)      # = log N(x|in) - log N(x|out), shared var


# ----------------------------------------------------------------------------- metrics
def auc(m: np.ndarray, n: np.ndarray) -> float:
    r = rankdata(np.concatenate([m, n]))
    return float((r[: len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def tpr_at_fpr(m: np.ndarray, n: np.ndarray, fpr: float = FPR) -> float:
    return float(np.mean(m > np.percentile(n, 100 * (1 - fpr))))     # strict '>': ties are misses


def bootstrap(m, n, rng, B: int = BOOT_N):
    a, t = np.empty(B), np.empty(B)
    for b in range(B):
        mi = m[rng.randint(0, len(m), len(m))]
        ni = n[rng.randint(0, len(n), len(n))]
        a[b], t[b] = auc(mi, ni), tpr_at_fpr(mi, ni)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(a), q(t)


def canary_percentile_median(m_raw: np.ndarray, n_raw: np.ndarray) -> float:
    """Median over members of the % of held-out raw scores strictly below the member's score (0-100)."""
    return float(np.median(np.searchsorted(np.sort(n_raw), m_raw, side="left") / len(n_raw) * 100.0))


def metric_block(m_cal, n_cal, m_raw, n_raw, seed_offset: int = 0) -> dict:
    rng = np.random.RandomState(SEED + seed_offset)
    ci_c, tci_c = bootstrap(m_cal, n_cal, rng)
    ci_r, _ = bootstrap(m_raw, n_raw, rng)
    return {
        "attack_auc": auc(m_cal, n_cal), "auc_ci95": ci_c,
        "tpr_at_1pct_fpr": tpr_at_fpr(m_cal, n_cal), "canary_percentile_median": canary_percentile_median(m_raw, n_raw),
        # extras (allowed by spec)
        "tpr_ci95": tci_c, "attack_auc_raw": auc(m_raw, n_raw), "auc_ci95_raw": ci_r,
        "tpr_at_1pct_fpr_raw": tpr_at_fpr(m_raw, n_raw),
    }


def _level(ss, level, members, nonmembers, shard_id, sh, seed_offset) -> dict:
    for df, name in ((members, "members"), (nonmembers, "nonmembers")):
        missing = [int(p) for p in df["person_id"] if int(p) not in sh["index"]]
        if missing:
            raise ValueError(f"{name}: {len(missing)} ids not in shadow canary pool (e.g. {missing[:3]}). "
                             "The shadow cache must be built BEFORE train.csv is rewritten by an erase.")
    m_raw = target_scores(ss, members, level, shard_id)
    n_raw = target_scores(ss, nonmembers, level, shard_id)
    S = sh["ens"] if level == "ensemble" else sh["shd"]
    mi = np.array([sh["index"][int(p)] for p in members["person_id"]])
    ni = np.array([sh["index"][int(p)] for p in nonmembers["person_id"]])
    m_cal = llr_scores(m_raw, mi, S, sh["inmask"])
    n_cal = llr_scores(n_raw, ni, S, sh["inmask"])
    blk = metric_block(m_cal, n_cal, m_raw, n_raw, seed_offset)
    blk["n_members"], blk["n_nonmembers"] = int(len(members)), int(len(nonmembers))
    return blk


# ----------------------------------------------------------------------------- public API
def run_audit(ss, members: pd.DataFrame, nonmembers: pd.DataFrame,
              shard_id: int | None = None, data_dir: str = DATA_DIR,
              n_workers: int = -1) -> dict:
    """Spec 8.3/8.4. shard_id=None -> ensemble block only (shard=None).
    With shard_id, members are restricted to member canaries whose shard_of == shard_id; the shard
    block uses that shard's own model, the ensemble block uses the deployed ensemble.

    Shard-level calibration note: held-out canaries are scored by the audited shard's model but
    calibrated with the in/out distributions of the shard that canary maps to (the held-out set is
    never in any shard, so this is the only well-defined 'in' reference)."""
    sh = get_shadows(ss, data_dir, n_jobs=n_workers)
    if shard_id is not None:
        keep = members["person_id"].map(lambda p: shard_of(int(p), ss.n_shards)) == shard_id
        members = members[keep.values]
    ens = _level(ss, "ensemble", members, nonmembers, None, sh, 1)
    out = {"ensemble": ens, "shard": None, "shard_detail": None}
    if shard_id is not None:
        shd = _level(ss, "shard", members, nonmembers, shard_id, sh, 2)
        out["shard"] = shd
        out["shard_detail"] = {"shard_id": int(shard_id), "n_members": int(len(members)),
                               "n_nonmembers": int(len(nonmembers)),
                               "attack_auc_raw": shd["attack_auc_raw"], "auc_ci95_raw": shd["auc_ci95_raw"],
                               "ensemble_attack_auc_raw": ens["attack_auc_raw"]}
    return out
