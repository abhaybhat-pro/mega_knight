"""Acceptance tests for src/report/narrate.py (8.6).

Acceptance: validate_numbers passes on the template report, and fails NAMING the number
when one number that is not in the receipt is injected.
The receipt below is a hand-made fixture with the 8.4 shape. It is NOT a measured result.
"""
import copy
import json
import os
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from src.report import narrate  # noqa: E402
from src.report.narrate import (generate_report_with_info, granite_report,  # noqa: E402
                                template_report, validate_numbers)

H = lambda c: c * 64  # noqa: E731


def sample_receipt() -> dict:
    return {
        "schema_version": 1, "request_id": "REQ-0001",
        "timestamp_utc": "2026-10-17T16:10:12Z",
        "persons_erased": 6, "shard_ids": [3], "rows_removed": 6,
        "model_hash_before": {"3": "3a9f12c4e7b1" + H("a")[12:]},
        "model_hash_after": {"3": "9d41e0b27c55" + H("b")[12:]},
        "full_retrain_seconds": 41.237, "shard_retrain_seconds": 5.3041, "speedup": 7.774,
        "test_auroc_before": 0.8712, "test_auroc_after": 0.8695,
        "audit_before": {
            "ensemble": {"attack_auc": 0.6123, "auc_ci95": [0.5811, 0.6432],
                         "tpr_at_1pct_fpr": 0.0425, "canary_percentile_median": 0.7311},
            "shard": {"attack_auc": 0.9012, "auc_ci95": [0.8806, 0.9217],
                      "tpr_at_1pct_fpr": 0.4125, "canary_percentile_median": 0.9544}},
        "audit_after": {
            "ensemble": {"attack_auc": 0.5021, "auc_ci95": [0.4713, 0.5330],
                         "tpr_at_1pct_fpr": 0.0100, "canary_percentile_median": 0.5102},
            "shard": {"attack_auc": 0.5133, "auc_ci95": [0.4820, 0.5444],
                      "tpr_at_1pct_fpr": 0.0125, "canary_percentile_median": 0.5207}},
        "guard": {"worst_subgroup": "village_code=3", "max_subgroup_auroc_drop": 0.0112,
                  "drift_flags": 11, "held_decisions": 482},
        "log_seq": 1, "log_entry_hash": "c0ffee" + H("1")[6:], "hmac": H("f"),
    }


@pytest.fixture
def rec():
    return sample_receipt()


# ------------------------------------------------------------------ acceptance
def test_template_report_passes(rec):
    text = template_report(rec)
    ok, bad = validate_numbers(text, rec)
    assert ok, f"unmatched numbers in template report: {bad}"
    assert bad == []


def test_first_line_names_writer(rec):
    assert template_report(rec).splitlines()[0].startswith("Writer: template")


@pytest.mark.parametrize("injected", ["7777", "0.9999", "13", "123456789"])
def test_injected_number_fails_and_is_named(rec, injected):
    text = template_report(rec) + f"\nAn extra finding: {injected} people were affected.\n"
    ok, bad = validate_numbers(text, rec)
    assert not ok
    assert injected in bad


def test_changed_figure_in_place_is_caught(rec):
    text = template_report(rec).replace("5.30 seconds", "5.31 seconds")
    ok, bad = validate_numbers(text, rec)
    assert not ok and "5.31" in bad


def test_injected_hash_prefix_fails(rec):
    text = template_report(rec) + "\nFingerprint deadbeef1234 was seen.\n"
    ok, bad = validate_numbers(text, rec)
    assert not ok and "deadbeef1234" in bad


def test_wrong_request_id_fails(rec):
    text = template_report(rec).replace("REQ-0001", "REQ-0002")
    ok, bad = validate_numbers(text, rec)
    assert not ok and "REQ-0002" in bad


# ------------------------------------------------------------------ normalisation rules
def test_rounding_and_formats_accepted(rec):
    ok, bad = validate_numbers("AUROC 0.87 and 0.870 and 0.8712; 7.8 times; 41 s? no", rec)
    assert not ok and bad == ["41"]  # 41 is not an integer in the receipt (41.237)
    ok, bad = validate_numbers("AUROC 0.87, speed 7.8 times, 41.24 seconds, 6 people", rec)
    assert ok, bad


def test_percent_form_accepted_only_with_percent_sign(rec):
    assert validate_numbers("AUROC was 87.1% before", rec)[0]
    ok, bad = validate_numbers("AUROC was 87.1 before", rec)
    assert not ok and bad == ["87.1"]


def test_exact_tie_values_match_whatever_the_rounding(rec):
    # found by the demo run: 0.0195 -> 1.95 is an exact tie at one decimal
    rec["audit_after"]["ensemble"]["tpr_at_1pct_fpr"] = 0.0195
    rec["test_auroc_after"] = 0.7775
    t = template_report(rec)
    assert "1.9%" not in t and "2.0%" in t       # decimal half-up, same on every CPU
    assert "0.778" in t
    assert validate_numbers(t, rec)[0]
    assert validate_numbers("it was 1.9% and 0.777", rec)[0]   # float-style rounding also OK
    assert not validate_numbers("it was 1.8%", rec)[0]


def test_thousands_separator(rec):
    rec["rows_removed"] = 12345
    assert validate_numbers("12,345 rows removed", rec)[0]


def test_integers_do_not_match_by_rounding(rec):
    # 0.87 rounds to 1 at zero decimals, but a bare "1" must not match through that
    rec2 = copy.deepcopy(rec)
    rec2["log_seq"] = 2
    ok, bad = validate_numbers("count 5", rec2)
    assert not ok and bad == ["5"]


def test_allowlist_and_writer_line(rec):
    ok, _ = validate_numbers("Digital Personal Data Protection Act, 2023", rec)
    assert ok
    ok, _ = validate_numbers("Writer: model x-9000 v77\nSummary 6 people", rec)
    assert ok  # first line is skipped


def test_zero_and_null_receipt_do_not_crash():
    empty = {"schema_version": 1, "guard": None, "audit_before": {}, "audit_after": {}}
    text = template_report(empty)
    ok, bad = validate_numbers(text, empty)
    assert ok, bad
    assert "not recorded" in text


def test_receipt_with_null_guard(rec):
    rec["guard"] = None
    text = template_report(rec)
    assert validate_numbers(text, rec)[0]
    assert "not recorded" in text


def test_template_covers_dpdp_sections_and_legal_note(rec):
    t = template_report(rec)
    for h in ("Erasure (DPDP", "Consent withdrawal (DPDP", "Integrity (DPDP"):
        assert h in t
    low = t.lower()
    assert "not legal advice" in low and "unsettled" in low and "to be verified" in low
    assert "not a mathematical proof" in low


def test_template_does_not_leak_full_hashes_or_hmac(rec):
    t = template_report(rec)
    assert rec["hmac"] not in t
    assert rec["model_hash_before"]["3"] not in t


# ------------------------------------------------------------------ granite plumbing
# These tests use a FAKE llama.cpp script to test our subprocess/validation/fallback code.
# They say nothing about whether real Granite runs: that stays Planned until it does.
def _fake_cli(tmp_path, body: str) -> str:
    p = tmp_path / "fake-llama-cli"
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture
def gguf(tmp_path):
    g = tmp_path / "fake.gguf"
    g.write_bytes(b"x")
    return str(g)


def test_granite_unconfigured_falls_back(rec, tmp_path, monkeypatch):
    monkeypatch.delenv("FORGETME_LLAMA_CLI", raising=False)
    monkeypatch.delenv("FORGETME_GRANITE_GGUF", raising=False)
    log = tmp_path / "g.log"
    text, info = generate_report_with_info(rec, log_path=log)
    assert info["writer"] == "template"
    assert text.splitlines()[0].startswith("Writer: template (fallback: Granite not used")
    assert validate_numbers(text, rec)[0]
    assert "UNAVAILABLE" in log.read_text()


@pytest.mark.linux_only
def test_granite_good_output_is_accepted(rec, tmp_path, gguf, monkeypatch):
    cli = _fake_cli(tmp_path, "echo 'Summary: 6 people were erased; AUROC 0.87.'")
    monkeypatch.setenv("FORGETME_LLAMA_CLI", cli)
    monkeypatch.setenv("FORGETME_GRANITE_GGUF", gguf)
    text, info = generate_report_with_info(rec, log_path=tmp_path / "g.log")
    assert info["writer"] == "granite"
    assert text.splitlines()[0].startswith("Writer: IBM Granite via llama.cpp")
    assert "6 people" in text


@pytest.mark.linux_only
def test_granite_invented_number_is_rejected_and_named(rec, tmp_path, gguf, monkeypatch):
    cli = _fake_cli(tmp_path, "echo 'Summary: 6 people erased, 7777 records at risk.'")
    monkeypatch.setenv("FORGETME_LLAMA_CLI", cli)
    monkeypatch.setenv("FORGETME_GRANITE_GGUF", gguf)
    log = tmp_path / "g.log"
    text, info = generate_report_with_info(rec, log_path=log)
    assert info["writer"] == "template"
    assert info["unmatched"] == ["7777"]
    assert "7777" in text.splitlines()[0]  # fallback is announced on line 1
    assert "7777" not in "\n".join(text.splitlines()[1:])
    assert "REJECTED" in log.read_text()


@pytest.mark.linux_only
def test_granite_nonzero_exit_falls_back(rec, tmp_path, gguf, monkeypatch):
    cli = _fake_cli(tmp_path, "echo boom >&2; exit 3")
    monkeypatch.setenv("FORGETME_LLAMA_CLI", cli)
    monkeypatch.setenv("FORGETME_GRANITE_GGUF", gguf)
    text, info = generate_report_with_info(rec, log_path=tmp_path / "g.log")
    assert info["writer"] == "template" and "exited with code" in info["fallback_reason"]


def test_granite_report_raises_when_unconfigured(rec, monkeypatch):
    monkeypatch.delenv("FORGETME_LLAMA_CLI", raising=False)
    with pytest.raises(narrate.GraniteUnavailable):
        granite_report(rec)


def test_prompt_contains_only_instructions_and_receipt(rec):
    p = narrate.build_prompt(rec)
    assert "ONLY the numbers" in p
    assert json.dumps(rec["request_id"]) in p
