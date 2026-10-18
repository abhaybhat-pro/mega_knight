"""Acceptance tests for bench/run_benchmarks.py (8.6).

Acceptance: run_benchmarks writes bench/results.csv with the required columns and the
VM's nproc. These tests run a tiny smoke configuration; they check the harness, not speed.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from bench import run_benchmarks as rb  # noqa: E402
from src.contracts import SEED, TRAIN_CSV  # noqa: E402
from src.data.generate import generate, read  # noqa: E402

_C = {}


def _data():
    if not _C:
        d = tempfile.mkdtemp()
        generate(out_dir=d)
        _C["dir"] = d
    return _C["dir"]


def _small():
    df = read(TRAIN_CSV, _data())
    return df.sample(n=1200, random_state=SEED).sort_values("person_id")


def test_worker_counts_cover_spec_and_core_count():
    assert rb.worker_counts(1) == [1, 2, 4, 8]
    assert rb.worker_counts(4) == [1, 2, 4, 8]
    assert rb.worker_counts(12) == [1, 2, 4, 8, 12]
    assert rb.worker_counts(16) == [1, 2, 4, 8, 16]
    assert rb.worker_counts(32) == [1, 2, 4, 8, 16, 32]


@pytest.mark.linux_only
def test_nproc_matches_available_cores():
    from src.runtime.resources import available_cores

    assert rb.get_nproc() == available_cores()
    try:
        reported = int(subprocess.run(["nproc"], capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        return
    # nproc and the affinity mask can disagree inside a long joblib session.
    # The column we publish is available_cores().
    if reported != available_cores():
        pytest.skip(f"nproc reported {reported} while available_cores is {available_cores()}")


def test_median_is_the_median_of_three_runs():
    calls = []
    import time
    delays = iter([0.03, 0.01, 0.02])

    def fn():
        calls.append(1)
        time.sleep(next(delays))

    m = rb.median_seconds(fn, 3)
    assert len(calls) == 3
    assert 0.018 < m < 0.05  # the 0.02 run, not the mean of the others or the min


def test_results_csv_schema_and_cores():
    out = tempfile.mkdtemp()
    data = _data()
    held = read("heldout_canaries.csv", data)
    rows = rb.run_benchmarks(_small(), out, repeats=1, workers=[1, 2],
                             shard_counts=(1, 4), quick=True, log=lambda s: None,
                             heldout_df=held, data_dir=data, cold_audit=False)
    path = os.path.join(out, "results.csv")
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames == ["experiment", "n_workers", "n_shards", "seconds", "n_cores"]
        got = list(rd)
    assert got and len(got) == len(rows)
    names = {r["experiment"] for r in got}
    # warm-cache audit rows must never carry the scaling name
    assert {"erase_vs_full", "train_scaling", "audit_warm_cache",
            "full_retrain_parallel", "full_retrain_serial"} <= names
    assert "audit_scaling" not in names
    nproc = rb.get_nproc()
    for r in got:
        assert int(r["n_cores"]) == nproc
        assert float(r["seconds"]) > 0
        workers = int(r["n_workers"])
        assert workers == -1 or workers >= 1
        assert int(r["n_shards"]) >= 1
    # the spec'd shard counts appear for the erase experiment
    assert {int(r["n_shards"]) for r in got if r["experiment"] == "erase_vs_full"} >= {1, 4, 8}


def test_meta_and_notes_state_cores_and_stand_in():
    out = tempfile.mkdtemp()
    rb.run_benchmarks(_small(), out, repeats=1, workers=[1, 2], shard_counts=(1,),
                      quick=True, log=lambda s: None)
    meta = json.load(open(os.path.join(out, "results_meta.json")))
    assert meta["n_cores"] == rb.get_nproc() and meta["quick"] is True
    assert "source" in meta["experiments"]["audit_scaling"]
    notes = open(os.path.join(out, "scaling_notes.md")).read()
    assert f"{rb.get_nproc()} core" in notes
    assert "QUICK SMOKE RUN" in notes


def test_cold_audit_scaling_uses_a_fresh_empty_models_dir_per_run(monkeypatch):
    """audit_scaling must time the shadow build, so each run gets its own empty cache dir."""
    seen = []

    def fake_audit(ss, members, heldout, shard_id=None, n_workers=1, data_dir=None, **kw):
        seen.append((ss.models_dir, os.path.isdir(ss.models_dir) and not os.listdir(ss.models_dir), n_workers))
        return {}

    monkeypatch.setattr(rb, "_real_audit_callable", lambda: (fake_audit, "fake"))
    out = tempfile.mkdtemp()
    data = _data()
    rb.run_benchmarks(_small(), out, repeats=1, workers=[1, 2], shard_counts=(1,), quick=True,
                      experiments=("audit_scaling",), log=lambda s: None,
                      heldout_df=read("heldout_canaries.csv", data), data_dir=data, audit_repeats=2)
    assert seen and all(empty for _d, empty, _w in seen)
    assert len({d for d, _e, _w in seen}) == len(seen)   # never reused
    meta = json.load(open(os.path.join(out, "results_meta.json")))
    assert meta["experiments"]["audit_scaling"]["cold_cache"] is True
    assert meta["experiments"]["audit_scaling"]["repeats"] == 2


def test_amdahl_estimate_recovers_known_serial_fraction():
    f = 0.2
    t = {w: f + (1 - f) / w for w in (1, 2, 4, 8)}
    est = rb.amdahl_serial_fraction(t, n_cores=8)
    assert abs(est - f) < 1e-9
    # workers above the core count are ignored
    t[16] = 5.0
    assert abs(rb.amdahl_serial_fraction(t, n_cores=8) - f) < 1e-9
    assert rb.amdahl_serial_fraction({1: 1.0}, n_cores=8) is None


def test_single_core_note_does_not_pretend_to_scale():
    rows = [{"experiment": "train_scaling", "n_workers": w, "n_shards": 8,
             "seconds": 10.0 + w, "n_cores": 1} for w in (1, 2, 4, 8)]
    meta = {"n_cores": 1, "machine": "x", "python": "3", "byteorder": "little",
            "n_train_rows": 10, "repeats": 3, "quick": False, "experiments": {}}
    note = rb.scaling_notes(rows, meta)
    assert "one core" in note and "oversubscription" in note


def test_main_quick_end_to_end():
    out = tempfile.mkdtemp()
    rc = rb.main(["--quick", "--repeats", "1", "--out", out, "--workers", "1,2",
                  "--data-dir", _data()])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "results.csv"))


def test_main_without_data_is_a_clear_error():
    assert rb.main(["--quick", "--data-dir", tempfile.mkdtemp()]) == 2
