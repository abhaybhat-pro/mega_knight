"""Facade aliases match the underlying calls and the documented shapes."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import FEATURES, ID_COL, LABEL_COL, N_SHARDS, SEED, ShardSet
from src.model.shards import train_all, train_shard


def _tiny(n=80, n_shards=2, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURES})
    df["age"] = rng.integers(18, 80, n)
    df["village_code"] = rng.integers(0, 8, n)
    df[LABEL_COL] = rng.integers(0, 2, n)
    df[ID_COL] = np.arange(100, 100 + n)
    df["canary_type"] = "none"
    return df


def test_make_data_ids_come_from_canary_type():
    from src.forgetme_api import make_data

    df, ids = make_data()
    assert list(ids) == ["member", "heldout"]
    assert ids["member"] == [int(v) for v in df.loc[df.canary_type == "member", "person_id"]]
    assert len(ids["heldout"]) == len(set(ids["heldout"]))
    assert set(ids["member"]).isdisjoint(ids["heldout"])


def test_train_shards_matches_train_all(tmp_path, monkeypatch):
    from src import forgetme_api as api

    df = _tiny()
    seen = {}
    real = api.train_all

    def wrapped(frame, n_shards=N_SHARDS, n_workers=-1, models_dir=None, seed=SEED, probe_X=None):
        seen["n_shards"] = n_shards
        seen["seed"] = seed
        return real(frame, n_shards=n_shards, n_workers=1,
                    models_dir=str(tmp_path / "models"), seed=seed, probe_X=probe_X)

    monkeypatch.setattr(api, "train_all", wrapped)
    out = api.train_shards(df, n_shards=2, seed=7)
    direct = train_all(df, n_shards=2, seed=7, n_workers=1, models_dir=str(tmp_path / "models2"))
    assert seen == {"n_shards": 2, "seed": 7}
    assert isinstance(out, ShardSet)
    assert out.models[0].predict(df[FEATURES].head(4)).tolist() == direct.models[0].predict(
        df[FEATURES].head(4)).tolist()


def _tmpdir():
    import tempfile
    return tempfile.mkdtemp()


def test_predict_proba_matches_predict(monkeypatch):
    from src import forgetme_api as api
    from src.model.shards import predict

    df = _tiny()
    ss = train_all(df, n_shards=2, seed=SEED, models_dir=_tmpdir())
    monkeypatch.setattr(api, "load", lambda: ss)
    got = api.predict_proba(df)
    assert np.allclose(got, predict(ss, df)["proba"])


def test_erase_delegates_and_hides_patient_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "facade-test-key")
    import fakes_erase as F
    from src import forgetme_api as api
    from src.erase.queue import erase as erase_flow

    env = F.make_env(tmp_path)
    pid = F.persons_in_shard(env, 1, 1, member=False)[0]
    monkeypatch.setattr(api, "load", lambda: env.ss)
    monkeypatch.setattr(api, "load_train_df", lambda: env.train)

    seen = {}

    def _flow(person_ids, request_id, ss, train_df, guard_summary=None):
        seen["guard_summary"] = guard_summary
        return erase_flow(person_ids, request_id, ss, train_df,
                          data_dir=env.data_dir, log_path=env.log_path,
                          measure_full_retrain=False, guard_summary=guard_summary)

    monkeypatch.setattr(api, "erase_flow", _flow)
    out = api.erase([pid], "REQ-FACADE")
    assert seen["guard_summary"] is None                 # not supplied: the receipt says "not recorded"
    assert out["receipt"]["guard"]["drift_flags"] is None
    assert set(out) == {"timings", "hashes", "rows_removed", "receipt"}
    assert out["rows_removed"] == out["receipt"]["rows_removed"]
    assert out["hashes"]["after"] == out["receipt"]["model_hash_after"]
    blob = str(out)
    assert "hba1c" not in blob and "needs_referral" not in blob


def test_audit_shape_matches_run_audit(monkeypatch):
    from src import forgetme_api as api

    raw = {"ensemble": {"attack_auc": 0.7, "tpr_at_1pct_fpr": 0.1,
                        "canary_percentile_median": 80.0, "auc_ci95": [0.6, 0.8]},
           "shard": None}
    monkeypatch.setattr(api, "run_audit", lambda *a, **k: raw)
    got = api.audit(object(), object(), object())
    assert got["auc"] == 0.7 and got["tpr_at_1fpr"] == 0.1
    assert got["canary_pct"] == 80.0 and got["ci"] == [0.6, 0.8]
    assert got["ensemble"]["auc"] == 0.7 and got["shard"] is None


def test_subgroup_delta_matches_compare():
    from src.forgetme_api import subgroup_delta
    from src.guard.monitor import compare_subgroup_auroc

    before = {"village": {"3": 0.9}}
    after = {"village": {"3": 0.8}}
    assert subgroup_delta(before, after) == compare_subgroup_auroc(before, after)


def test_make_receipt_matches_build(monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "facade-test-key")
    from receipt_fixtures import pre_receipt_fields
    from src.forgetme_api import make_receipt
    from src.report.receipt import build_receipt

    fields = pre_receipt_fields()
    assert make_receipt(**fields) == build_receipt(**fields)


def test_verify_chain_true_then_false(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "facade-test-key")
    from receipt_fixtures import KEY, pre_receipt_fields
    from src.forgetme_api import verify_chain
    from src.report.receipt import build_receipt, finalize_receipt

    path = str(tmp_path / "log.jsonl")
    final = finalize_receipt(build_receipt(**pre_receipt_fields()), KEY, path)
    assert verify_chain(path) is True
    assert verify_chain(path, final) is True
    final["persons_erased"] = 99
    assert verify_chain(path, final) is False


def test_shard_of_reexport():
    from src.forgetme_api import shard_of as facade_shard
    from src.model.shards import shard_of

    assert facade_shard is shard_of
    assert facade_shard(1_000_003, N_SHARDS) == shard_of(1_000_003, N_SHARDS)


def test_printable_receipt_has_verification_and_no_features(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGETME_HMAC_KEY", "facade-test-key")
    from receipt_fixtures import KEY, pre_receipt_fields
    from src.report.receipt import build_receipt, finalize_receipt, receipt_to_html, receipt_to_text

    log = str(tmp_path / "log.jsonl")
    final = finalize_receipt(build_receipt(**pre_receipt_fields()), KEY, log)
    text = receipt_to_text(final)
    assert "not legal advice" in text.lower()
    assert "HMAC-SHA256" in text
    assert final["request_id"] in text
    assert "hba1c" not in text
    html = receipt_to_html(final)
    assert html.startswith("<!DOCTYPE html>")
    assert "<pre>" in html
