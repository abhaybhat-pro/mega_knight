"""Test scaffolding for the erase module.

src/model/shards.py, src/audit/mia.py and src/guard/monitor.py are the real modules. Importing this file FIRST makes any
of them that cannot be imported yet fall back to a small spec-conformant fake
(registered in sys.modules only; no file under src/ is created). Once the real
modules exist they are used automatically and nothing here is substituted.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.contracts import FEATURES, N_SHARDS, SEED, ShardSet

USING_FAKE: dict[str, bool] = {}


# ----------------------------------------------------------------- fakes
def _fake_shards() -> types.ModuleType:
    m = types.ModuleType("src.model.shards")

    def shard_of(person_id, n_shards=N_SHARDS):
        return int(hashlib.sha256(str(int(person_id)).encode()).hexdigest(), 16) % n_shards

    def train_shard(df_shard, shard_id, seed=SEED):
        df = df_shard.sort_values("person_id", kind="mergesort")
        rf = RandomForestClassifier(n_estimators=25, max_depth=None, min_samples_leaf=1,
                                    bootstrap=True, n_jobs=1, random_state=seed + shard_id)
        return rf.fit(df[FEATURES], df["needs_referral"])

    def fingerprint(model, probe_X):
        p = np.round(model.predict_proba(probe_X[FEATURES])[:, 1], 8).astype(">f8")
        return hashlib.sha256(p.tobytes()).hexdigest()

    def predict(ss, X):
        per = np.stack([ss.models[k].predict_proba(X[FEATURES])[:, 1] for k in sorted(ss.models)])
        p = per.mean(axis=0)
        eps = 1e-12
        ent = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
        return {"proba": p, "label": (p >= 0.5).astype(int),
                "confidence": np.maximum(p, 1 - p), "entropy": ent, "per_shard": per}

    m.shard_of, m.train_shard, m.fingerprint, m.predict = shard_of, train_shard, fingerprint, predict
    return m


def _fake_mia() -> types.ModuleType:
    m = types.ModuleType("src.audit.mia")

    def _true_label_score(proba, y):
        return np.where(np.asarray(y) == 1, proba, 1 - proba)

    def _view(sm, sn, rng):
        y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]
        s = np.r_[sm, sn]
        auc = float(roc_auc_score(y, s))
        boots = []
        for _ in range(50):
            i = rng.integers(0, len(sm), len(sm))
            j = rng.integers(0, len(sn), len(sn))
            boots.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[sm[i], sn[j]]))
        thr = np.percentile(sn, 99)
        return {"attack_auc": auc, "auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "tpr_at_1pct_fpr": float((sm > thr).mean()),
                "canary_percentile_median": float(np.median([(sn <= v).mean() for v in sm]))}

    def run_audit(ss, members, nonmembers, shard_id=None):
        sh = importlib.import_module("src.model.shards")
        rng = np.random.default_rng(0)
        ens_m = _true_label_score(sh.predict(ss, members[FEATURES])["proba"], members["needs_referral"])
        ens_n = _true_label_score(sh.predict(ss, nonmembers[FEATURES])["proba"], nonmembers["needs_referral"])
        if shard_id is None:
            sm, sn = ens_m, ens_n
        else:
            mod = ss.models[shard_id]
            sm = _true_label_score(mod.predict_proba(members[FEATURES])[:, 1], members["needs_referral"])
            sn = _true_label_score(mod.predict_proba(nonmembers[FEATURES])[:, 1], nonmembers["needs_referral"])
        return {"ensemble": _view(ens_m, ens_n, rng), "shard": _view(sm, sn, rng)}

    m.run_audit = run_audit
    return m


def _fake_monitor() -> types.ModuleType:
    m = types.ModuleType("src.guard.monitor")

    def subgroup_auroc(ss, test_df):
        sh = importlib.import_module("src.model.shards")
        p = sh.predict(ss, test_df[FEATURES])["proba"]
        y = test_df["needs_referral"].to_numpy()
        out = {}
        for v in range(8):
            idx = (test_df["village_code"] == v).to_numpy()
            if idx.sum() >= 20 and len(set(y[idx])) == 2:
                out[f"village_{v}"] = float(roc_auc_score(y[idx], p[idx]))
        for name, lo, hi in (("age_18-39", 18, 39), ("age_40-59", 40, 59), ("age_60+", 60, 200)):
            idx = test_df["age"].between(lo, hi).to_numpy()
            if idx.sum() >= 20 and len(set(y[idx])) == 2:
                out[name] = float(roc_auc_score(y[idx], p[idx]))
        return out

    m.subgroup_auroc = subgroup_auroc
    return m


def _install(modname: str, parent: str, attr: str, factory) -> None:
    try:
        importlib.import_module(modname)
        USING_FAKE[modname] = False
        return
    except ImportError:
        pass
    fake = factory()
    sys.modules[modname] = fake
    pkg = sys.modules.get(parent)
    if pkg is None:
        pkg = types.ModuleType(parent)
        pkg.__path__ = []  # mark as package
        sys.modules[parent] = pkg
    setattr(pkg, attr, fake)
    USING_FAKE[modname] = True


_install("src.model.shards", "src.model", "shards", _fake_shards)
_install("src.audit.mia", "src.audit", "mia", _fake_mia)
_install("src.guard.monitor", "src.guard", "monitor", _fake_monitor)


# ----------------------------------------------------------------- data
def _features(rng, n, extreme=False):
    age = rng.integers(18, 91, n)
    d = {
        "age": age,
        "sex": rng.integers(0, 2, n),
        "bmi": rng.normal(27, 5, n).clip(15, 45),
        "sbp": rng.normal(125, 15, n),
        "dbp": rng.normal(80, 10, n),
        "hba1c": rng.normal(5.8, 0.9, n).clip(4, 12),
        "smoker": rng.integers(0, 2, n),
        "sym_fatigue": rng.integers(0, 2, n),
        "sym_vision": rng.integers(0, 2, n),
        "sym_thirst": rng.integers(0, 2, n),
        "sym_chest_pain": rng.integers(0, 2, n),
        "village_code": rng.integers(0, 8, n),
        "prior_visits": rng.poisson(2, n),
    }
    if extreme:
        d["age"] = rng.integers(18, 25, n)
        d["bmi"] = rng.uniform(40, 55, n)
        d["hba1c"] = rng.uniform(11, 14, n)
        d["sbp"] = rng.uniform(150, 200, n)
    return pd.DataFrame(d)


def make_dataset(seed=SEED, n_ordinary=1200, n_members=160, n_heldout=160, n_test=500):
    rng = np.random.default_rng(seed)
    ordv = _features(rng, n_ordinary)
    z = 0.04 * (ordv.age - 50) + 0.08 * (ordv.bmi - 27) + 1.2 * (ordv.hba1c - 5.8) + rng.normal(0, 1, n_ordinary)
    ordv["needs_referral"] = (z > 0).astype(int)
    ordv["person_id"] = np.arange(1, n_ordinary + 1)
    ordv["canary_type"] = "none"
    mem = _features(rng, n_members, extreme=True)
    mem["needs_referral"] = rng.integers(0, 2, n_members)
    mem["person_id"] = 1_000_000 + np.arange(n_members)
    mem["canary_type"] = "member"
    held = _features(rng, n_heldout, extreme=True)
    held["needs_referral"] = rng.integers(0, 2, n_heldout)
    held["person_id"] = 2_000_000 + np.arange(n_heldout)
    held["canary_type"] = "heldout"
    test = _features(rng, n_test)
    zt = 0.04 * (test.age - 50) + 0.08 * (test.bmi - 27) + 1.2 * (test.hba1c - 5.8) + rng.normal(0, 1, n_test)
    test["needs_referral"] = (zt > 0).astype(int)
    test["person_id"] = 3_000_000 + np.arange(n_test)
    test["canary_type"] = "none"
    cols = ["person_id"] + FEATURES + ["needs_referral", "canary_type"]
    train = pd.concat([ordv, mem], ignore_index=True)[cols]
    return train, test[cols], held[cols]


def build_shardset(train_df, models_dir, n_shards=N_SHARDS):
    sh = importlib.import_module("src.model.shards")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    col = train_df["person_id"].map(lambda p: sh.shard_of(int(p), n_shards))
    models = {}
    for k in range(n_shards):
        models[k] = sh.train_shard(train_df[col == k], k)
        joblib.dump(models[k], models_dir / f"shard_{k}.joblib")
    manifest = {"n_shards": n_shards, "shards": {str(k): {"file": f"shard_{k}.joblib"} for k in range(n_shards)}}
    (models_dir / "manifest.json").write_text(json.dumps(manifest))
    return ShardSet(models=models, n_shards=n_shards, models_dir=str(models_dir), manifest=manifest)


def make_env(tmp_path) -> SimpleNamespace:
    tmp_path = Path(tmp_path)
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    train, test, held = make_dataset()
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    held.to_csv(data / "heldout_canaries.csv", index=False)
    ss = build_shardset(train, tmp_path / "models")
    sh = importlib.import_module("src.model.shards")
    shard = train["person_id"].map(lambda p: sh.shard_of(int(p), ss.n_shards))
    return SimpleNamespace(
        ss=ss, train=train, test=test, held=held, shard_col=shard,
        data_dir=str(data), log_path=str(tmp_path / "logs" / "erasure_log.jsonl"),
        models_dir=Path(tmp_path / "models"), root=tmp_path, sh=sh,
    )


def persons_in_shard(env, k, n, member=None):
    df = env.train[env.shard_col == k]
    if member is True:
        df = df[df.canary_type == "member"]
    elif member is False:
        df = df[df.canary_type == "none"]
    ids = [int(x) for x in df["person_id"].head(n)]
    assert len(ids) == n, f"not enough persons in shard {k}"
    return ids
