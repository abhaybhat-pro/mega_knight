"""Acceptance tests for src/data/generate.py (8.6)."""
import hashlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.contracts import (CANARY_COL, COLUMNS, HELDOUT_CSV, ID_BASE, ID_COL,  # noqa: E402
                           LABEL_COL, N_CANARY, N_PATIENTS, N_STREAM_BASE, N_TEST,
                           STREAM_CSV, TEST_CSV, THRESHOLDS, TRAIN_CSV)
from src.data.generate import drift_stream, generate, is_out_of_range  # noqa: E402

FILES = (TRAIN_CSV, TEST_CSV, HELDOUT_CSV, STREAM_CSV)
_CACHE = {}


def _dirs():
    if not _CACHE:
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        generate(out_dir=a)
        generate(out_dir=b)
        _CACHE["a"], _CACHE["b"] = a, b
    return _CACHE["a"], _CACHE["b"]


def _sha(d, f):
    with open(os.path.join(d, f), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _read(f):
    return pd.read_csv(os.path.join(_dirs()[0], f))


def test_two_runs_byte_identical():
    a, b = _dirs()
    for f in FILES:
        assert _sha(a, f) == _sha(b, f), f


def test_row_counts_and_columns():
    for f, n in ((TRAIN_CSV, N_PATIENTS), (TEST_CSV, N_TEST),
                 (HELDOUT_CSV, N_CANARY), (STREAM_CSV, N_STREAM_BASE)):
        df = _read(f)
        assert len(df) == n, f
        assert list(df.columns) == COLUMNS, f
        assert df[ID_COL].is_unique, f


def test_canary_counts_and_id_ranges():
    tr, he = _read(TRAIN_CSV), _read(HELDOUT_CSV)
    mem = tr[tr[CANARY_COL] == "member"]
    assert len(mem) == N_CANARY and (he[CANARY_COL] == "heldout").all()
    assert mem[ID_COL].min() == ID_BASE["member"]
    assert he[ID_COL].min() == ID_BASE["heldout"]
    assert set(tr[CANARY_COL]) == {"none", "member"}
    assert tr[ID_COL].is_monotonic_increasing


def test_no_heldout_id_in_train_and_no_overlap_between_files():
    tr, he, te, st = (_read(f) for f in (TRAIN_CSV, HELDOUT_CSV, TEST_CSV, STREAM_CSV))
    ids = [set(d[ID_COL]) for d in (tr, he, te, st)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (ids[i] & ids[j]), (i, j)


def test_canary_labels_roughly_balanced():
    tr, he = _read(TRAIN_CSV), _read(HELDOUT_CSV)
    for df in (tr[tr[CANARY_COL] == "member"], he):
        assert 0.4 <= df[LABEL_COL].mean() <= 0.6


def test_canaries_out_of_range_and_ordinary_rows_are_not():
    tr, he, te, st = (_read(f) for f in (TRAIN_CSV, HELDOUT_CSV, TEST_CSV, STREAM_CSV))
    assert is_out_of_range(tr[tr[CANARY_COL] == "member"]).all()
    assert is_out_of_range(he).all()
    for df in (tr[tr[CANARY_COL] == "none"], te, st):
        assert not is_out_of_range(df).any()


def test_schema_ranges():
    tr = _read(TRAIN_CSV)
    assert tr["age"].between(18, 90).all()
    assert set(tr["sex"]) <= {0, 1} and set(tr["smoker"]) <= {0, 1}
    for c in ("sym_fatigue", "sym_vision", "sym_thirst", "sym_chest_pain"):
        assert set(tr[c]) <= {0, 1}
    assert tr["village_code"].between(0, 7).all() and (tr["prior_visits"] >= 0).all()
    assert set(tr[LABEL_COL]) == {0, 1}


def test_village3_share_and_batch_size():
    st = _read(STREAM_CSV)
    assert 0.22 <= (st["village_code"] == 3).mean() <= 0.28
    sizes = [(b["village_code"] == 3).sum() for b in drift_stream(st, shift=0)]
    assert 100 <= np.mean(sizes) <= 150  # about 125 per 500-row batch


def test_drift_stream_shift_applies_to_one_village_from_start_batch():
    st = _read(STREAM_CSV)
    batches = list(drift_stream(st))
    assert len(batches) == THRESHOLDS["n_batches"]
    assert all(len(b) == THRESHOLDS["batch_size"] for b in batches)
    for i, b in enumerate(batches):
        raw = st.iloc[i * 500:(i + 1) * 500].reset_index(drop=True)
        v3 = raw["village_code"] == 3
        if i < 10:
            pd.testing.assert_frame_equal(b, raw)
        else:
            assert np.allclose(b.loc[v3, "hba1c"], raw.loc[v3, "hba1c"] + 1.5)
            assert (b.loc[~v3, "hba1c"] == raw.loc[~v3, "hba1c"]).all()
            other = [c for c in raw.columns if c != "hba1c"]
            pd.testing.assert_frame_equal(b[other], raw[other])


def test_no_drift_stream_for_zero_shift_or_late_start():
    st = _read(STREAM_CSV)
    plain = [st.iloc[i * 500:(i + 1) * 500].reset_index(drop=True) for i in range(20)]
    for kw in ({"shift": 0}, {"start_batch": 25}):
        got = list(drift_stream(st, **kw))
        assert len(got) == 20
        for g, p in zip(got, plain):
            pd.testing.assert_frame_equal(g, p)
