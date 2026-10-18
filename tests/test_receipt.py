"""FR-6: receipt verifies; any value change / single byte flip fails; no patient-level data."""
import copy
import json

import pytest

from receipt_fixtures import KEY, make_pre_receipt, pre_receipt_fields
from src.report import receipt as R


def signed():
    r = make_pre_receipt()
    r["log_seq"] = 1
    r["log_entry_hash"] = "c" * 64
    r["hmac"] = R.sign(r, KEY)
    return r


def mutate(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.125
    if isinstance(v, str):
        return v + "x"
    if v is None:
        return 0
    if isinstance(v, list):
        return v + [0]
    if isinstance(v, dict):
        return {**v, "zz": 1}
    raise AssertionError(type(v))


def test_verify_true():
    assert R.verify_receipt(signed(), KEY) is True


def test_sign_is_hmac_sha256_over_canonical_json_without_hmac():
    import hashlib
    import hmac as _h
    r = signed()
    body = {k: v for k, v in r.items() if k != "hmac"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert r["hmac"] == _h.new(KEY, raw, hashlib.sha256).hexdigest()


def test_wrong_key_fails():
    assert R.verify_receipt(signed(), b"another-key") is False


def test_any_top_level_value_change_fails():
    r = signed()
    for k in [k for k in r if k != "hmac"]:
        t = copy.deepcopy(r)
        t[k] = mutate(t[k])
        assert R.verify_receipt(t, KEY) is False, f"change to {k} not detected"


def test_nested_value_change_fails():
    r = signed()
    t = copy.deepcopy(r)
    t["audit_after"]["shard"]["attack_auc"] = 0.5
    assert R.verify_receipt(t, KEY) is False
    t = copy.deepcopy(r)
    t["model_hash_after"]["3"] = "d" * 64
    assert R.verify_receipt(t, KEY) is False
    t = copy.deepcopy(r)
    t["audit_before"]["ensemble"]["auc_ci95"][0] = 0.0
    assert R.verify_receipt(t, KEY) is False


def test_removed_added_or_missing_hmac_fails():
    r = signed()
    for k in [k for k in r if k != "hmac"]:
        t = {x: v for x, v in r.items() if x != k}
        assert R.verify_receipt(t, KEY) is False, f"removal of {k} not detected"
    t = {**r, "extra": 1}
    assert R.verify_receipt(t, KEY) is False
    t = {x: v for x, v in r.items() if x != "hmac"}
    assert R.verify_receipt(t, KEY) is False
    assert R.verify_receipt({**r, "hmac": 123}, KEY) is False


def test_hmac_case_change_fails():
    r = signed()
    flipped = r["hmac"].upper() if r["hmac"] != r["hmac"].upper() else r["hmac"].lower()
    assert R.verify_receipt({**r, "hmac": flipped}, KEY) is False


def test_every_single_byte_flip_fails():
    raw = R.receipt_bytes(signed())
    assert R.verify_receipt_bytes(raw, KEY) is True
    for i in range(len(raw)):
        for mask in (0x01, 0x20, 0x80):
            bad = bytearray(raw)
            bad[i] ^= mask
            assert R.verify_receipt_bytes(bytes(bad), KEY) is False, f"byte {i} mask {mask:#x}"


def test_byte_flip_detected_even_when_json_still_parses():
    raw = R.receipt_bytes(signed())
    detected_by_dict_level = 0
    for i in range(len(raw)):
        bad = bytearray(raw)
        bad[i] ^= 0x01
        try:
            parsed = json.loads(bytes(bad).decode("utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            assert R.verify_receipt(parsed, KEY) is False
            detected_by_dict_level += 1
    assert detected_by_dict_level > 0


def test_duplicate_keys_and_trailing_bytes_rejected():
    raw = R.receipt_bytes(signed())
    assert R.verify_receipt_bytes(raw + b" ", KEY) is False
    assert R.verify_receipt_bytes(raw[:-1] + b',"schema_version":1}', KEY) is False


def test_missing_key_env_raises(monkeypatch):
    monkeypatch.delenv(R.HMAC_ENV, raising=False)
    with pytest.raises(RuntimeError, match="FORGETME_HMAC_KEY"):
        R.get_hmac_key()
    monkeypatch.setenv(R.HMAC_ENV, "")
    with pytest.raises(RuntimeError):
        R.get_hmac_key()
    monkeypatch.setenv(R.HMAC_ENV, "k")
    assert R.get_hmac_key() == b"k"


def test_no_default_key_on_sign_and_verify():
    with pytest.raises(ValueError):
        R.sign(make_pre_receipt(), b"")
    with pytest.raises(ValueError):
        R.verify_receipt(signed(), None)


# ---- no patient-level data -------------------------------------------------
def test_builds_valid_receipt_with_only_whitelisted_keys():
    r = make_pre_receipt()
    assert set(r) <= R.ALLOWED_KEYS
    assert r["schema_version"] == 1


@pytest.mark.parametrize("bad", [
    {"person_id": 1000001},
    {"person_ids": [1000001, 1000002]},
    {"age": 22},
    {"patients": []},
    {"canary_type": "member"},
    {"anything_else": 1},
])
def test_top_level_patient_or_unknown_fields_rejected(bad):
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(**bad))


def test_nested_patient_fields_rejected():
    f = pre_receipt_fields()
    f["audit_before"] = {"shard": {"attack_auc": 0.9, "person_id": 5}}
    with pytest.raises(ValueError):
        R.build_receipt(**f)
    f = pre_receipt_fields(shard_detail={"3": {"age": 20}})
    with pytest.raises(ValueError):
        R.build_receipt(**f)


def test_person_ids_cannot_hide_in_shard_ids_or_lists():
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(shard_ids=[1000001], model_hash_before={"1000001": "a" * 64},
                                             model_hash_after={"1000001": "b" * 64}))
    f = pre_receipt_fields()
    f["audit_before"] = {"shard": {"attack_auc": 0.9, "scores": list(range(1000000, 1000100))}}
    with pytest.raises(ValueError):
        R.build_receipt(**f)


def test_free_text_in_metric_blocks_rejected():
    f = pre_receipt_fields()
    f["audit_before"] = {"shard": {"attack_auc": 0.9, "who": "Jane Doe"}}
    with pytest.raises(ValueError):
        R.build_receipt(**f)
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(guard={"worst_subgroup": "Jane Doe, village 3"}))
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(guard={"held_decisions": [1000001, 1000002]}))
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(note="patient 1000001 erased"))
    with pytest.raises(ValueError):  # trailing newline must not slip past the regex
        R.build_receipt(**pre_receipt_fields(note="erased\n"))
    with pytest.raises(ValueError):
        R.build_receipt(**pre_receipt_fields(request_id="REQ-0001\n"))


def test_schema_violations_rejected():
    for over in (dict(request_id="has space"), dict(persons_erased=-1), dict(rows_removed=1.5),
                 dict(test_auroc_after=1.5), dict(speedup=float("nan")),
                 dict(model_hash_after={"3": "short"}), dict(model_hash_after={"4": "b" * 64}),
                 dict(timestamp_utc="yesterday"), dict(schema_version=2)):
        with pytest.raises(ValueError):
            R.build_receipt(**pre_receipt_fields(**over))
    f = pre_receipt_fields()
    del f["audit_before"]
    with pytest.raises(ValueError):
        R.build_receipt(**f)


def test_null_audit_blocks_with_note_and_default_guard_are_valid():
    f = pre_receipt_fields(audit_before=None, audit_after=None,
                           note="No member canaries among erased persons; audit not applicable")
    del f["guard"]
    r = R.build_receipt(**f)
    assert r["audit_before"] is None and r["guard"]["worst_subgroup"] is None


def test_numpy_values_are_converted():
    import numpy as np
    r = R.build_receipt(**pre_receipt_fields(
        persons_erased=np.int64(6), speedup=np.float64(7.5), shard_ids=np.array([3])))
    assert type(r["persons_erased"]) is int and type(r["speedup"]) is float
    json.dumps(r)
