"""Acceptance tests for src/model/shards.py (8.6)."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.contracts import (CANARY_COL, FEATURES, HELDOUT_CSV, ID_COL, LABEL_COL,  # noqa: E402
                           MANIFEST_NAME, N_PATIENTS, N_SHARDS, PERSON_MAP_NAME,
                           TEST_CSV, TRAIN_CSV)
from src.data.generate import generate, read  # noqa: E402
from src.model import shards  # noqa: E402

_C = {}


def _env():
    if not _C:
        d = tempfile.mkdtemp()
        generate(out_dir=d)
        _C["data"] = d
        _C["train"] = read(TRAIN_CSV, d)
        _C["held"] = read(HELDOUT_CSV, d)
        _C["probe"] = shards.load_probe(d)
        for w in (1, 4):
            m = tempfile.mkdtemp()
            _C[f"ss{w}"] = shards.train_all(_C["train"], n_workers=w, models_dir=m,
                                            probe_X=_C["probe"])
    return _C


def test_shard_of_identical_across_processes():
    ids = [1, 2, 3, 100000, 1000000, 2000123, 3999999]
    here = [shards.shard_of(i) for i in ids]
    code = ("import sys,json;sys.path.insert(0,%r);from src.model.shards import shard_of;"
            "print(json.dumps([shard_of(i) for i in %r]))" % (ROOT, ids))
    for seed in ("0", "123"):  # different PYTHONHASHSEED: built-in hash() would differ
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, check=True).stdout
        assert json.loads(out) == here
    assert all(0 <= s < N_SHARDS for s in here)


def test_each_person_in_exactly_one_shard():
    e = _env()
    tr = e["train"]
    sid = shards.assign_shards(tr[ID_COL])
    parts = [set(tr.loc[sid == k, ID_COL]) for k in range(N_SHARDS)]
    assert sum(len(p) for p in parts) == N_PATIENTS
    assert len(set().union(*parts)) == N_PATIENTS  # disjoint and complete
    man = e["ss1"].manifest
    assert [man["shards"][str(k)]["rows"] for k in range(N_SHARDS)] == [len(p) for p in parts]
    assert man["total_rows"] == N_PATIENTS


def test_member_canaries_in_training_and_heldout_never():
    e = _env()
    tr, he = e["train"], e["held"]
    sid = shards.assign_shards(tr[ID_COL])
    members = tr[tr[CANARY_COL] == "member"]
    msid = shards.assign_shards(members[ID_COL])
    for k in range(N_SHARDS):
        in_shard = set(tr.loc[sid == k, ID_COL])
        assert set(members.loc[msid == k, ID_COL]) <= in_shard
    assert len(set(msid)) == N_SHARDS  # member canaries spread over every shard
    assert not (set(he[ID_COL]) & set(tr[ID_COL]))
    assert (he[CANARY_COL] == "heldout").all()


def test_files_and_manifest_written():
    e = _env()
    d = e["ss1"].models_dir
    names = sorted(os.listdir(d))
    assert names == sorted([f"shard_{k}.joblib" for k in range(N_SHARDS)]
                           + [MANIFEST_NAME, PERSON_MAP_NAME])  # no temp leftovers
    man = json.load(open(os.path.join(d, MANIFEST_NAME)))
    assert man["n_shards"] == N_SHARDS and len(man["shards"]) == N_SHARDS
    assert all(len(v["model_hash"]) == 64 for v in man["shards"].values())
    pmap = json.load(open(os.path.join(d, PERSON_MAP_NAME)))
    assert len(pmap) == N_PATIENTS


def test_one_worker_vs_four_workers_identical_hashes():
    e = _env()
    a, b = e["ss1"].manifest, e["ss4"].manifest
    assert a["shards"] == b["shards"]
    assert a["ensemble_hash"] == b["ensemble_hash"]


def test_row_order_does_not_change_hashes():
    e = _env()
    shuffled = e["train"].sample(frac=1.0, random_state=7)
    ss = shards.train_all(shuffled, n_workers=1, models_dir=tempfile.mkdtemp(),
                          probe_X=e["probe"])
    assert ss.manifest["shards"] == e["ss1"].manifest["shards"]


def test_load_roundtrip_matches_trained():
    e = _env()
    loaded = shards.load(e["ss1"].models_dir)
    assert loaded.n_shards == N_SHARDS
    te = read(TEST_CSV, e["data"])
    assert np.array_equal(shards.predict(loaded, te)["proba"],
                          shards.predict(e["ss1"], te)["proba"])


def test_owning_shard_has_memorised_member_canaries_but_not_heldout():
    # Sanity check (not an 8.6 number): the audit depends on this gap existing.
    e = _env()
    ss = e["ss1"]

    def acc(df):
        sid = shards.assign_shards(df[ID_COL])
        hit = [ss.models[k].predict(df.iloc[[i]][FEATURES])[0] == df.iloc[i][LABEL_COL]
               for i, k in enumerate(sid)]
        return float(np.mean(hit))
    mem = e["train"][e["train"][CANARY_COL] == "member"]
    assert acc(mem) > acc(e["held"]) + 0.2
