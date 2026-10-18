"""FR-4 erase, FR-12 subgroup check, atomic swap, fail-fast behaviour."""
import hashlib
import json
import sys
import threading
from pathlib import Path

import joblib
import pytest

import fakes_erase as F  # must come before importing src.erase.queue (installs fakes if needed)
from receipt_fixtures import KEY
from src.contracts import FEATURES, THRESHOLDS
from src.erase import queue as Q
from src.model import shards
from src.report import receipt as R


pytestmark = pytest.mark.slow  # trains many forests; excluded from `make test`

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(R.HMAC_ENV, KEY.decode())
    return F.make_env(tmp_path)


@pytest.fixture
def trains(monkeypatch):
    calls = []
    orig = shards.train_shard

    def counting(df, k, seed=None, **kw):
        calls.append(int(k))
        return orig(df, k) if seed is None else orig(df, k, seed)

    monkeypatch.setattr(shards, "train_shard", counting)
    return calls


def probe_of(env):
    return env.test[FEATURES].head(int(THRESHOLDS.get("probe_rows", 1000)))


def fp(env, model):
    return shards.fingerprint(model, probe_of(env))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_erase(env, ids, rid="REQ-0001", **kw):
    return Q.erase(ids, rid, env.ss, env.train, data_dir=env.data_dir, log_path=env.log_path, **kw)


def all_files(d):
    """Served files in the models directory. The shadow-score cache is a subdirectory."""
    return sorted(p.name for p in Path(d).iterdir() if p.is_file())


# ---------------------------------------------------------------- FR-4
def test_only_shard_k_changes_and_result_is_exact(env, trains):
    pid = F.persons_in_shard(env, 3, 1, member=True)[0]
    before = {k: fp(env, m) for k, m in env.ss.models.items()}
    files_before = {k: sha_file(env.models_dir / f"shard_{k}.joblib") for k in range(8)}
    train_csv_before = sha_file(Path(env.data_dir) / "train.csv")

    r = run_erase(env, [pid])

    assert r["shard_ids"] == [3] and r["persons_erased"] == 1 and r["rows_removed"] == 1
    assert trains == [3]  # exactly one real retrain, shard 3 only

    after = {k: fp(env, m) for k, m in env.ss.models.items()}
    assert after[3] != before[3]
    for k in range(8):
        if k != 3:
            assert after[k] == before[k], f"shard {k} changed"
            assert sha_file(env.models_dir / f"shard_{k}.joblib") == files_before[k]
    assert r["model_hash_before"] == {"3": before[3]}
    assert r["model_hash_after"] == {"3": after[3]}

    # exactness: fresh train_shard on shard 3's data minus the person (original, uncounted fn)
    remaining = env.train[(env.shard_col == 3) & (env.train.person_id != pid)]
    fresh = Q._REAL_TRAIN_SHARD(remaining, 3)
    assert fp(env, fresh) == after[3] == r["model_hash_after"]["3"]

    # on disk: new model file loads and matches the receipt
    assert fp(env, joblib.load(env.models_dir / "shard_3.joblib")) == after[3]
    manifest = json.loads((env.models_dir / "manifest.json").read_text())
    assert manifest["shards"]["3"]["model_hash"] == after[3]
    assert "hash" not in manifest["shards"]["3"]

    # rows gone; train.csv untouched; train_current.csv has everyone else
    cur = Q.load_train_df(env.data_dir)
    assert pid not in set(cur.person_id) and len(cur) == len(env.train) - 1
    assert sha_file(Path(env.data_dir) / "train.csv") == train_csv_before

    # receipt integrity + linkage + chain
    assert R.verify_receipt(r, KEY)
    assert R.verify_receipt_log_linkage(r, env.log_path) == (True, "linked")
    assert R.verify_log(env.log_path) == (True, None)
    assert r["log_seq"] == 1


def test_two_ids_in_same_shard_cause_exactly_one_retrain(env, trains):
    a, b = F.persons_in_shard(env, 5, 2, member=True)
    r = run_erase(env, [a, b])
    assert trains == [5]
    assert r["persons_erased"] == 2 and r["rows_removed"] == 2 and r["shard_ids"] == [5]
    cur = Q.load_train_df(env.data_dir)
    assert not ({a, b} & set(cur.person_id))


def test_duplicate_ids_in_request_count_once(env, trains):
    a = F.persons_in_shard(env, 2, 1, member=True)[0]
    r = run_erase(env, [a, a, a])
    assert r["persons_erased"] == 1 and trains == [2]


def test_scratch_full_retrain_is_timed_but_never_swapped_in(env, trains):
    pid = F.persons_in_shard(env, 1, 1, member=False)[0]
    live_before = {k: id(m) for k, m in env.ss.models.items()}
    r = run_erase(env, [pid])
    assert r["full_retrain_seconds"] > 0 and r["shard_retrain_seconds"] > 0
    assert r["speedup"] == pytest.approx(r["full_retrain_seconds"] / r["shard_retrain_seconds"])
    assert trains == [1]  # scratch retrain does not count as (or become) the served retrain
    for k, m in env.ss.models.items():
        if k != 1:
            assert id(m) == live_before[k]
    assert all_files(env.models_dir) == sorted([f"shard_{k}.joblib" for k in range(8)] + ["manifest.json"])


def test_multi_shard_receipt_uses_highest_before_auc_shard(env, trains):
    a = F.persons_in_shard(env, 0, 3, member=True)
    b = F.persons_in_shard(env, 6, 3, member=True)
    r = run_erase(env, a + b)
    assert sorted(trains) == [0, 6]
    assert r["shard_ids"] == [0, 6]
    assert set(r["shard_detail"]) == {"0", "6"}
    best = max(r["shard_detail"].values(), key=lambda d: d["audit_before"]["shard"]["attack_auc"])
    assert r["audit_before"] == best["audit_before"] and r["audit_after"] == best["audit_after"]
    assert all(d["members_audited"] == 3 for d in r["shard_detail"].values())
    assert set(r["model_hash_before"]) == set(r["model_hash_after"]) == {"0", "6"}
    assert R.verify_receipt(r, KEY)


def test_audit_after_runs_on_new_models(env):
    pid = F.persons_in_shard(env, 4, 8, member=True)
    r = run_erase(env, pid)
    b, a = r["audit_before"]["shard"], r["audit_after"]["shard"]
    assert b["attack_auc"] > a["attack_auc"]   # flow sanity: 'after' used the retrained shard
    assert r["test_auroc_before"] > 0.5 and r["test_auroc_after"] > 0.5


def test_no_member_canaries_gives_null_audit_blocks_with_note(env, trains):
    pid = F.persons_in_shard(env, 7, 1, member=False)[0]
    r = run_erase(env, [pid])
    assert r["audit_before"] is None and r["audit_after"] is None
    assert "No member canaries" in r["note"]
    assert trains == [7] and R.verify_receipt(r, KEY)


def test_unknown_ids_produce_logged_receipt_without_retrain(env, trains):
    r = run_erase(env, [987654321])
    assert trains == [] and r["persons_erased"] == 0 and r["shard_ids"] == [] and r["persons_not_found"] == 1
    assert R.verify_receipt(r, KEY) and R.verify_receipt_log_linkage(r, env.log_path)[0]
    assert not (Path(env.data_dir) / "train_current.csv").exists()


def test_erased_ids_do_not_come_back_in_later_retrains(env, trains):
    a, b = F.persons_in_shard(env, 3, 2, member=True)
    run_erase(env, [a])
    cur = Q.load_train_df(env.data_dir)
    r2 = Q.erase([b], "REQ-0002", env.ss, cur, data_dir=env.data_dir, log_path=env.log_path)
    final = Q.load_train_df(env.data_dir)
    assert not ({a, b} & set(final.person_id))
    assert r2["log_seq"] == 2 and R.verify_log(env.log_path) == (True, None)
    remaining = env.train[(env.shard_col == 3) & ~env.train.person_id.isin([a, b])]
    assert r2["model_hash_after"]["3"] == fp(env, Q._REAL_TRAIN_SHARD(remaining, 3))


def test_receipt_and_log_hold_no_patient_content(env):
    ids = F.persons_in_shard(env, 3, 3, member=True)
    r = run_erase(env, ids)
    blob = json.dumps(r) + Path(env.log_path).read_text()
    for pid in ids:
        assert str(pid) not in blob
    assert set(r) <= R.ALLOWED_KEYS
    for col in ("person_id", "needs_referral", "canary_type"):
        assert col not in blob


# ---------------------------------------------------------------- atomicity
def test_failure_while_writing_swap_leaves_everything_unchanged(env, monkeypatch, trains):
    a = F.persons_in_shard(env, 0, 1, member=True)
    b = F.persons_in_shard(env, 6, 1, member=True)
    files_before = {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)}
    live_before = dict(env.ss.models)
    real_dump = joblib.dump
    n_swap_dumps = []

    def flaky(obj, path, *args, **kw):
        if Path(path).name.startswith(".shard_"):
            n_swap_dumps.append(path)
            if len(n_swap_dumps) == 2:           # first temp written, second one dies
                raise OSError("disk full (simulated)")
        return real_dump(obj, path, *args, **kw)

    monkeypatch.setattr(Q.joblib, "dump", flaky)
    with pytest.raises(OSError):
        run_erase(env, a + b)
    assert len(n_swap_dumps) == 2
    assert {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)} == files_before  # no tmp left
    assert all(env.ss.models[k] is live_before[k] for k in live_before)
    assert not (Path(env.data_dir) / "train_current.csv").exists()
    assert not Path(env.log_path).exists()


def test_retrain_failure_changes_nothing(env, monkeypatch):
    pid = F.persons_in_shard(env, 2, 1, member=True)
    files_before = {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)}

    orig = shards.train_shard

    def boom(*a, **k):
        raise RuntimeError("train failed")

    monkeypatch.setattr(shards, "train_shard", boom)
    with pytest.raises(RuntimeError):
        run_erase(env, pid)
    assert {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)} == files_before
    assert not (Path(env.data_dir) / "train_current.csv").exists()
    monkeypatch.setattr(shards, "train_shard", orig)
    r = run_erase(env, pid)   # retry works because train_current was never written
    assert r["persons_erased"] == 1


def test_readers_never_see_a_partial_model_file(env):
    ids = F.persons_in_shard(env, 3, 3, member=True)
    path = env.models_dir / "shard_3.joblib"
    stop, errors, loads = threading.Event(), [], [0]

    def reader():
        while not stop.is_set():
            try:
                m = joblib.load(path)
                m.predict_proba(probe_of(env).head(5))
                loads[0] += 1
            except Exception as exc:  # any half-written file would land here
                errors.append(repr(exc))

    t = threading.Thread(target=reader)
    t.start()
    try:
        train = env.train
        for i, pid in enumerate(ids):
            Q.erase([pid], f"REQ-{i:04d}", env.ss, train, data_dir=env.data_dir, log_path=env.log_path)
            train = Q.load_train_df(env.data_dir)
    finally:
        stop.set()
        t.join()
    assert errors == [] and loads[0] > 0


# ---------------------------------------------------------------- fail fast
def test_missing_key_fails_before_touching_anything(env, monkeypatch):
    monkeypatch.delenv(R.HMAC_ENV)
    pid = F.persons_in_shard(env, 3, 1, member=True)
    files_before = {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)}
    with pytest.raises(RuntimeError, match="FORGETME_HMAC_KEY"):
        run_erase(env, pid)
    assert {n: sha_file(env.models_dir / n) for n in all_files(env.models_dir)} == files_before
    assert not (Path(env.data_dir) / "train_current.csv").exists()


@pytest.mark.parametrize("rid", ["bad id", "", "x" * 80, "REQ\n1"])
def test_invalid_request_id_fails_before_touching_anything(env, rid):
    pid = F.persons_in_shard(env, 3, 1, member=True)
    with pytest.raises(ValueError):
        run_erase(env, pid, rid=rid)
    assert not (Path(env.data_dir) / "train_current.csv").exists()


def test_non_integer_ids_rejected(env):
    with pytest.raises(ValueError):
        run_erase(env, ["1000001"])
    with pytest.raises(ValueError):
        run_erase(env, [True])


def test_sloppy_audit_output_cannot_break_or_pollute_receipt(env, monkeypatch):
    real = sys.modules["src.audit.mia"].run_audit

    def messy(ss, members, nonmembers, shard_id=None, **kwargs):
        out = real(ss, members, nonmembers, shard_id, **kwargs)
        out["shard"]["debug"] = "Jane Doe was here"
        out["shard"]["person_id"] = 1000000
        out["ensemble"]["scores"] = list(range(500))
        out["secondary"] = {"attack_auc": 0.5}
        return out

    monkeypatch.setattr(sys.modules["src.audit.mia"], "run_audit", messy)
    r = run_erase(env, F.persons_in_shard(env, 3, 2, member=True))
    blob = json.dumps(r)
    assert "Jane" not in blob and "person_id" not in blob
    assert R.verify_receipt(r, KEY)


# ---------------------------------------------------------------- guard (FR-12)
def test_guard_block_filled_with_subgroup_numbers(env):
    r = run_erase(env, F.persons_in_shard(env, 3, 3, member=True))
    g = r["guard"]
    assert g["worst_subgroup"] is not None and isinstance(g["max_subgroup_auroc_drop"], float)
    assert g["drift_flags"] is None and g["held_decisions"] is None
    assert isinstance(g["flagged_subgroups"], dict)


def test_subgroup_after_erase_lists_every_drop_above_threshold(env, monkeypatch):
    monitor = sys.modules["src.guard.monitor"]
    answers = iter([
        {"village_0": 0.80, "village_1": 0.80, "age_60+": 0.70, "age_18-39": 0.75},
        {"village_0": 0.75, "village_1": 0.79, "age_60+": 0.66, "age_18-39": 0.79},
    ])
    monkeypatch.setattr(monitor, "subgroup_auroc", lambda ss, df: next(answers))
    thr = THRESHOLDS.get("subgroup_auroc_drop", 0.03)
    assert thr == 0.03
    r = run_erase(env, F.persons_in_shard(env, 3, 2, member=True))
    g = r["guard"]
    assert g["worst_subgroup"] == "village_0"
    assert g["max_subgroup_auroc_drop"] == pytest.approx(0.05)
    assert set(g["flagged_subgroups"]) == {"village_0", "age_60+"}   # village_1 (0.01) is not listed
    assert g["flagged_subgroups"]["village_0"] == pytest.approx(0.05)
    assert R.verify_receipt(r, KEY)


def test_guard_nested_subgroup_output_is_flattened(env, monkeypatch):
    monitor = sys.modules["src.guard.monitor"]
    answers = iter([{"village": {3: 0.9, 4: 0.8}, "age_band": {"60+": 0.7}},
                    {"village": {3: 0.8, 4: 0.8}, "age_band": {"60+": 0.7}}])
    monkeypatch.setattr(monitor, "subgroup_auroc", lambda ss, df: next(answers))
    r = run_erase(env, F.persons_in_shard(env, 3, 1, member=True))
    assert r["guard"]["worst_subgroup"] == "village_3"
    assert set(r["guard"]["flagged_subgroups"]) == {"village_3"}


def test_guard_missing_gives_null_guard_block(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.guard.monitor", None)   # ImportError on import
    r = run_erase(env, F.persons_in_shard(env, 3, 1, member=True))
    assert r["guard"] == {"worst_subgroup": None, "max_subgroup_auroc_drop": None,
                          "drift_flags": None, "held_decisions": None}
    assert "Guard not available" in r["note"] and R.verify_receipt(r, KEY)


def test_guard_crash_does_not_lose_the_receipt(env, monkeypatch):
    monkeypatch.setattr(sys.modules["src.guard.monitor"], "subgroup_auroc",
                        lambda ss, df: (_ for _ in ()).throw(ValueError("boom")))
    r = run_erase(env, F.persons_in_shard(env, 3, 1, member=True))
    assert r["guard"]["worst_subgroup"] is None and "Guard subgroup check failed" in r["note"]
    assert R.verify_receipt(r, KEY)
