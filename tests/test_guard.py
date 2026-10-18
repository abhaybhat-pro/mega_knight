"""Guard drift, the no-drift stream, rare villages, and hold-list privacy."""
import json

import pandas as pd
import pytest

from src.contracts import THRESHOLDS
from src.data.generate import drift_stream, generate, read
from src.guard.monitor import (
    BatchReport, fit_baseline, public_counts, score_batch, stream_advisories, stream_counts,
)
from src.model.shards import train_all


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("guarddata")
    data = root / "data"
    models = root / "models"
    generate(out_dir=str(data))
    train = read("train.csv", str(data))
    test = read("test.csv", str(data))
    stream = read("stream_base.csv", str(data))
    ss = train_all(train, n_workers=-1, models_dir=str(models))
    return {"ss": ss, "train": train, "test": test, "stream": stream, "data": str(data)}


def _walk(ss, train, test, stream, shift):
    base = fit_baseline(train, ss=ss, test_df=test)
    reports = []
    batches = []
    for batch in drift_stream(stream, shift=shift):
        batches.append(batch)
        reports.append(score_batch(ss, batch, base))
    return reports, batches


def test_drift_flags_village_3_only_and_rare_villages_are_insufficient(trained, tmp_path):
    start = int(THRESHOLDS["drift_start_batch"])
    reports, batches = _walk(trained["ss"], trained["train"], trained["test"], trained["stream"],
                             shift=float(THRESHOLDS["drift_shift"]))
    first = None
    for i, report in enumerate(reports):
        holds = [row for row in report.subgroups if row["status"] == "hold-for-review"]
        if not holds:
            continue
        assert [row["label"] for row in holds] == ["village_3"]
        window = pd.concat(batches[max(0, i - 1): i + 1], ignore_index=True)
        held = set(report.held_ids)
        assert held
        got = set(window.loc[window["person_id"].isin(held), "village_code"].astype(int))
        assert got == {3}
        first = i
        break
    assert first is not None and first <= start + 1, first
    # Rare villages stay under 200 rows in a 2-batch window.
    rare = [row for row in reports[start + 1].subgroups if row["label"] in {"village_6", "village_7"}]
    assert rare and all(row["status"] == "insufficient data" and not row["flagged"] for row in rare)
    blob = json.dumps(public_counts(reports[first]))
    for pid in reports[first].held_ids:
        assert str(pid) not in blob
    path = tmp_path / "guard_counts.json"
    path.write_text(blob, encoding="utf-8")
    assert str(reports[first].held_ids[0]) not in path.read_text(encoding="utf-8")
    _assert_no_denial(reports)


def test_no_drift_stream_has_zero_holds_and_zero_village_flags(trained):
    """Window stays 2 (SC-9).

    A sweep on this rehearsal (windows 2, 3, 4, 6, 8) found no length that both
    holds village 3 by batch start+1 and raises zero advisories. At window 2 the
    no-drift stream has zero holds. Two age-band advisories sit just above
    psi_moderate (measured PSI about 0.10 to 0.11). Window 6 clears those
    advisories and moves the village-3 hold from batch 10 to batch 12.
    """
    reports, _batches = _walk(trained["ss"], trained["train"], trained["test"], trained["stream"], shift=0)
    assert len(reports) == int(THRESHOLDS["n_batches"])
    flagged = [
        (i, row["label"], row["status"], row["max_psi"])
        for i, report in enumerate(reports)
        for row in report.subgroups
        if row["flagged"]
    ]
    assert not any(row["status"] == "hold-for-review" for report in reports for row in report.subgroups)
    assert all(not label.startswith("village_") for _i, label, _status, _psi in flagged)
    assert all(status == "warning" for _i, _label, status, _psi in flagged)
    _assert_no_denial(reports)


def _row(label, kind, status, flagged=True):
    return {"label": label, "kind": kind, "n": 250, "status": status, "flagged": flagged,
            "mean_confidence": 0.9, "mean_entropy": 0.3, "max_psi": 0.3, "reason": "x"}


def test_stream_counts_are_counts_only_and_deduplicate_the_overlapping_window():
    """The 2-batch window overlaps: person 7 is held in two consecutive reports but counts once."""
    hold3 = _row("village_3", "village", "hold-for-review")
    adv = _row("age_60_plus", "age_band", "warning")
    reports = [
        BatchReport(subgroups=[hold3, adv], held_ids=[5, 6, 7], psi={}, entropy=0.2, outcome="hold-for-review", flags=[]),
        BatchReport(subgroups=[hold3], held_ids=[7, 8], psi={}, entropy=0.2, outcome="hold-for-review", flags=[]),
        BatchReport(subgroups=[_row("village_6", "village", "insufficient data", flagged=False)],
                    held_ids=[], psi={}, entropy=0.2, outcome="proceed", flags=[]),
    ]
    counts = stream_counts(reports)
    assert counts == {"drift_flags": 1, "held_decisions": 4}        # village_3 once; ids 5,6,7,8
    assert all(isinstance(v, int) for v in counts.values())
    assert stream_advisories(reports) == {"age_60_plus": 1}          # an advisory, never a hold
    assert stream_counts([]) == {"drift_flags": 0, "held_decisions": 0}


def test_stream_counts_on_the_real_streams(trained):
    """Drift stream: village 3 only. No-drift stream: zero holds, zero village flags."""
    drift, _ = _walk(trained["ss"], trained["train"], trained["test"], trained["stream"],
                     shift=float(THRESHOLDS["drift_shift"]))
    quiet, _ = _walk(trained["ss"], trained["train"], trained["test"], trained["stream"], shift=0)
    c = stream_counts(drift)
    assert c["drift_flags"] == 1 and c["held_decisions"] > 0
    assert stream_counts(quiet) == {"drift_flags": 0, "held_decisions": 0}
    assert all(not label.startswith("village_") for label in stream_advisories(quiet))


def _assert_no_denial(reports):
    allowed = {"proceed", "warn", "hold-for-review", "ok", "warning", "insufficient data"}
    for report in reports:
        assert report.outcome in {"proceed", "warn", "hold-for-review"}
        blob = json.dumps(public_counts(report)).lower()
        assert "deny" not in blob and "reject" not in blob
        for row in report.subgroups:
            assert row["status"] in allowed
