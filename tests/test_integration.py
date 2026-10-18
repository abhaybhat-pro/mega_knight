"""Cross-module checks for the merge lead. These do not replace any dev's tests.

The slow erase test trains real forests and builds the audit's shadow cache.
Everything else only imports modules or reads source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_shardset_is_the_contracts_class():
    import src.audit.mia as mia
    import src.contracts as contracts
    import src.erase.queue as queue
    import src.model.shards as shards

    assert queue.ShardSet is contracts.ShardSet
    assert shards.ShardSet is contracts.ShardSet
    # mia types ShardSet only through the object it is given; the class it would
    # construct is the same module everyone else imports.
    assert mia.shard_of is shards.shard_of


def test_stub_functions_are_not_exported_from_contracts():
    import src.contracts as contracts

    assert "erase" not in contracts.__all__
    assert "run_audit" not in contracts.__all__
    assert contracts.SEED == 42
    assert contracts.THRESHOLDS["subgroup_auroc_drop"] == 0.03
    assert contracts.THRESHOLDS["min_subgroup_size"] == 200


def test_log_functions_have_one_implementation():
    from src.erase.queue import append_log, verify_log
    from src.report.receipt import append_log as receipt_append
    from src.report.receipt import verify_log as receipt_verify

    assert append_log is receipt_append
    assert verify_log is receipt_verify


def test_served_retrain_goes_through_the_module_attribute():
    import inspect

    from src.erase import queue

    body = inspect.getsource(queue._erase_batch)
    assert "shards.train_shard(" in body
    scratch = inspect.getsource(queue._scratch_full_retrain)
    assert "_REAL_TRAIN_SHARD" in scratch


def test_shadow_cache_key_is_content_plus_seed_and_ignores_retrains(tmp_path):
    from src.audit.mia import shadow_cache_key
    from src.data.generate import generate
    from src.model.shards import train_shard
    import pandas as pd

    data = tmp_path / "data"
    generate(out_dir=str(data))
    key_a = shadow_cache_key(str(data), 8, seed=42)
    key_b = shadow_cache_key(str(data), 8, seed=42)
    key_c = shadow_cache_key(str(data), 8, seed=43)
    assert key_a == key_b and key_a != key_c
    train = pd.read_csv(data / "train.csv")
    before = shadow_cache_key(str(data), 8, seed=42)
    # A shard retrain does not rewrite train.csv, so the cache key stays put.
    train_shard(train.head(80), 0)
    after = shadow_cache_key(str(data), 8, seed=42)
    assert before == after
    flipped = tmp_path / "data2"
    flipped.mkdir()
    raw = (data / "train.csv").read_bytes()
    (flipped / "train.csv").write_bytes(raw[:-1] + (b"\n" if raw[-1:] != b"\n" else b"x"))
    (flipped / "heldout_canaries.csv").write_bytes((data / "heldout_canaries.csv").read_bytes())
    assert shadow_cache_key(str(flipped), 8, seed=42) != before


def test_receipt_log_linkage_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "integration-test-key")
    from receipt_fixtures import pre_receipt_fields
    from src.report import receipt as R
    from src.report.narrate import template_report, validate_numbers

    key = b"integration-test-key"
    log_path = str(tmp_path / "erasure_log.jsonl")
    final = R.finalize_receipt(R.build_receipt(**pre_receipt_fields()), key, log_path)
    assert R.verify_receipt(final, key)
    assert R.verify_log(log_path) == (True, None)
    ok, reason = R.verify_receipt_log_linkage(final, log_path)
    assert ok, reason

    flipped = dict(final)
    flipped["rows_removed"] = final["rows_removed"] + 1
    assert R.verify_receipt(flipped, key) is False

    raw = Path(log_path).read_bytes()
    Path(log_path).write_bytes(raw[:20] + bytes([raw[20] ^ 0x01]) + raw[21:])
    assert R.verify_log(log_path)[0] is False

    text = template_report(final)
    passed, bad = validate_numbers(text, final)
    assert passed, bad
    assert "template" in text.lower() or text.startswith("Writer:")


def test_patient_keys_stay_out_of_the_receipt_whitelist():
    from src.report.receipt import ALLOWED_KEYS, PATIENT_FIELD_DENYLIST

    assert not (set(ALLOWED_KEYS) & set(PATIENT_FIELD_DENYLIST))


def test_canary_identity_uses_the_column_and_the_published_bases():
    from src.contracts import CANARY_COL, ID_BASE

    source = (ROOT / "src" / "data" / "generate.py").read_text(encoding="utf-8")
    assert "CANARY_COL" in source
    assert CANARY_COL == "canary_type"
    assert ID_BASE["member"] == 1_000_000
    assert ID_BASE["heldout"] == 2_000_000
    assert ID_BASE["test"] == 3_000_000
    assert ID_BASE["stream"] == 4_000_000


def test_guard_module_is_on_disk():
    path = ROOT / "src" / "guard" / "monitor.py"
    assert path.is_file(), (
        "src/guard/monitor.py is missing. "
        "erase() catches that ImportError and leaves the guard block null. "
        "scripts/run_demo.sh exits 3 until this file exists, unless --no-guard is passed."
    )


def test_ui_does_not_ship_mock_numbers():
    hits = []
    app = ROOT / "app"
    for path in app.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "MOCK" in text or "import mock" in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], "Gate 2 forbids a MOCK UI: " + ", ".join(hits)


def test_benchmark_uses_the_current_training_set():
    text = (ROOT / "bench" / "run_benchmarks.py").read_text(encoding="utf-8")
    assert "train_current.csv" in text, (
        "bench/run_benchmarks.py reads train.csv only. After an erase the live "
        "training set is data/train_current.csv. Timing a full retrain on train.csv "
        "puts the erased rows back into the comparison."
    )


@pytest.mark.slow
def test_erase_updates_manifest_model_hash(tmp_path, monkeypatch):
    """One real erase. The manifest assertion fails while queue.py writes 'hash' and leaves 'model_hash' stale."""
    import json

    import pandas as pd

    monkeypatch.setenv("FORGETME_HMAC_KEY", "integration-test-key")
    import fakes_erase as F
    from src.erase.queue import erase
    from src.report import receipt as R

    env = F.make_env(tmp_path)
    train_before = Path(env.data_dir, "train.csv").read_bytes()
    env.ss.manifest["shards"]["3"]["model_hash"] = "0" * 64
    pid = F.persons_in_shard(env, 3, 1, member=True)[0]
    receipt = erase(
        [pid], "REQ-0001", env.ss, env.train,
        data_dir=env.data_dir, log_path=env.log_path, measure_full_retrain=False,
    )
    key = b"integration-test-key"
    assert R.verify_receipt(receipt, key)
    assert R.verify_log(env.log_path) == (True, None)
    linked, why = R.verify_receipt_log_linkage(receipt, env.log_path)
    assert linked, why
    assert Path(env.data_dir, "train.csv").read_bytes() == train_before
    current = pd.read_csv(Path(env.data_dir, "train_current.csv"))
    assert pid not in set(current["person_id"])
    for line in Path(env.log_path).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        assert "person_id" not in entry
    assert env.ss.manifest["shards"]["3"]["model_hash"] == receipt["model_hash_after"]["3"]
