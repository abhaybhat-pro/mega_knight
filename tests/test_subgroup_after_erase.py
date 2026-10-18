"""After an erase the receipt guard block is filled from subgroup AUROC."""
import json

import pytest

import fakes_erase as F
from src.contracts import THRESHOLDS
from src.erase.queue import erase
from src.guard.monitor import compare_subgroup_auroc
from src.report.receipt import verify_receipt


def test_compare_lists_only_drops_above_the_frozen_threshold():
    before = {"overall": 0.90, "village": {"3": 0.80, "1": 0.80}, "age_band": {"60_plus": 0.70}}
    after = {"overall": 0.90, "village": {"3": 0.70, "1": 0.79}, "age_band": {"60_plus": 0.66}}
    flagged = compare_subgroup_auroc(before, after, THRESHOLDS["subgroup_auroc_drop"])
    assert set(flagged) == {"village_3", "age_band_60_plus"}
    assert flagged["village_3"] == pytest.approx(0.10)


def test_erase_receipt_guard_block(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "subgroup-test-key")
    env = F.make_env(tmp_path)
    pid = F.persons_in_shard(env, 3, 1, member=False)[0]
    receipt = erase([pid], "REQ-GUARD", env.ss, env.train, data_dir=env.data_dir,
                    log_path=env.log_path, measure_full_retrain=False)
    guard = receipt["guard"]
    for key in ("worst_subgroup", "max_subgroup_auroc_drop", "drift_flags", "held_decisions"):
        assert key in guard
    assert isinstance(guard["flagged_subgroups"], dict)
    thr = THRESHOLDS["subgroup_auroc_drop"]
    for _label, drop in guard["flagged_subgroups"].items():
        assert drop > thr
    assert verify_receipt(receipt, b"subgroup-test-key")
    blob = json.dumps(receipt)
    assert "hba1c" not in blob and "person_id" not in blob and "needs_referral" not in blob


def _erase_with(tmp_path, monkeypatch, summary, rid):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "subgroup-test-key")
    env = F.make_env(tmp_path)
    pid = F.persons_in_shard(env, 3, 1, member=False)[0]
    return erase([pid], rid, env.ss, env.train, data_dir=env.data_dir, log_path=env.log_path,
                 measure_full_retrain=False, guard_summary=summary)


def test_guard_stream_counts_reach_the_receipt_as_counts_only(tmp_path, monkeypatch):
    receipt = _erase_with(tmp_path, monkeypatch, {"drift_flags": 1, "held_decisions": 37}, "REQ-GCOUNT")
    assert receipt["guard"]["drift_flags"] == 1 and receipt["guard"]["held_decisions"] == 37
    assert verify_receipt(receipt, b"subgroup-test-key")
    # schema keys are unchanged and nothing patient-level was added
    assert set(receipt["guard"]) == {"worst_subgroup", "max_subgroup_auroc_drop", "drift_flags",
                                     "held_decisions", "flagged_subgroups"}
    blob = json.dumps(receipt)
    assert "person_id" not in blob and "held_ids" not in blob


def test_receipt_guard_counts_are_null_without_a_guard_summary(tmp_path, monkeypatch):
    receipt = _erase_with(tmp_path, monkeypatch, None, "REQ-GNULL")
    assert receipt["guard"]["drift_flags"] is None and receipt["guard"]["held_decisions"] is None


@pytest.mark.parametrize("bad", [{"drift_flags": -1}, {"held_decisions": 2.5}, {"held_decisions": True},
                                 {"held_ids": [1, 2]}, [1, 2]])
def test_invalid_guard_summary_is_refused_before_anything_changes(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError):
        _erase_with(tmp_path, monkeypatch, bad, "REQ-GBAD")


def test_report_states_the_real_counts_and_the_validator_passes(tmp_path, monkeypatch):
    from src.report.narrate import template_report, validate_numbers
    receipt = _erase_with(tmp_path, monkeypatch, {"drift_flags": 1, "held_decisions": 37}, "REQ-GREP")
    text = template_report(receipt)
    assert "1 village reached the major drift line" in text
    assert "37 decisions were held for human review" in text
    ok, bad = validate_numbers(text, receipt)
    assert ok, bad
    (tmp_path / "n").mkdir()
    null_receipt = _erase_with(tmp_path / "n", monkeypatch, None, "REQ-GREP2")
    text2 = template_report(null_receipt)
    assert "not recorded" in text2 and validate_numbers(text2, null_receipt)[0]
