"""Acceptance tests for predict()/fingerprint() in src/model/shards.py (8.6)."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.contracts import N_SHARDS, TEST_CSV, TRAIN_CSV, THRESHOLDS  # noqa: E402
from src.data.generate import generate, read  # noqa: E402
from src.model import shards  # noqa: E402

_C = {}


def _env():
    if not _C:
        d = tempfile.mkdtemp()
        generate(out_dir=d)
        _C["test"] = read(TEST_CSV, d)
        _C["probe"] = shards.load_probe(d)
        _C["ss"] = shards.train_all(read(TRAIN_CSV, d), n_workers=1,
                                    models_dir=tempfile.mkdtemp(), probe_X=_C["probe"])
        _C["pred"] = shards.predict(_C["ss"], _C["test"])
    return _C


def test_keys_shapes_and_ranges():
    e = _env()
    p, n = e["pred"], len(e["test"])
    assert set(p) == {"proba", "label", "confidence", "entropy", "per_shard"}
    assert len(p["per_shard"]) == N_SHARDS
    for k in ("proba", "label", "confidence", "entropy"):
        assert p[k].shape == (n,)
    for arr in [p["proba"], p["confidence"], *p["per_shard"].values()]:
        assert np.all(arr >= 0) and np.all(arr <= 1)
    assert set(np.unique(p["label"])) <= {0, 1}
    assert np.all(p["entropy"] >= 0) and np.all(p["entropy"] <= 1 + 1e-12)


def test_ensemble_is_mean_of_per_shard():
    p = _env()["pred"]
    assert np.allclose(p["proba"], shards.per_shard_matrix(p).mean(axis=0), atol=1e-12)


def test_label_confidence_entropy_definitions():
    p = _env()["pred"]
    thr = THRESHOLDS["referral_threshold"]
    assert np.array_equal(p["label"], (p["proba"] >= thr).astype(int))
    assert np.allclose(p["confidence"], np.maximum(p["proba"], 1 - p["proba"]))
    q = p["proba"]
    ref = np.zeros_like(q)
    m = (q > 0) & (q < 1)
    ref[m] = -(q[m] * np.log2(q[m]) + (1 - q[m]) * np.log2(1 - q[m]))
    assert np.allclose(p["entropy"], ref, atol=1e-12)


def test_fingerprint_stable_and_sensitive():
    e = _env()
    ss, probe = e["ss"], e["probe"]
    assert len(probe) == THRESHOLDS["probe_rows"]
    h0 = shards.fingerprint(ss.models[0], probe)
    assert h0 == shards.fingerprint(ss.models[0], probe) and len(h0) == 64
    assert h0 != shards.fingerprint(ss.models[1], probe)
    assert h0 == ss.manifest["shards"]["0"]["model_hash"]
    assert shards.fingerprint(ss, probe) == ss.manifest["ensemble_hash"]
