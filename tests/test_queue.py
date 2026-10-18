"""FR-15: coalescing queue - cost scales with shards touched, not requests."""
import threading

import pytest

import fakes_erase as F  # noqa: F401  (installs fakes before the queue module imports shards)
from receipt_fixtures import KEY
from src.contracts import FEATURES, THRESHOLDS
from src.erase import queue as Q
from src.model import shards
from src.report import receipt as R

pytestmark = pytest.mark.slow  # trains many forests; excluded from `make test`

SHARDS = (1, 4, 6)


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


def new_queue(env, **kw):
    return Q.EraseQueue(env.ss, env.train, data_dir=env.data_dir, log_path=env.log_path, **kw)


def twenty_requests(env):
    """20 single-person requests spread over exactly 3 shards (7 + 7 + 6)."""
    ids = []
    for k, n in zip(SHARDS, (7, 7, 6)):
        ids += [(k, p) for p in F.persons_in_shard(env, k, n, member=True)]
    return [(f"REQ-{i:04d}", k, p) for i, (k, p) in enumerate(ids, start=1)]


def fps(env):
    probe = env.test[FEATURES].head(int(THRESHOLDS.get("probe_rows", 1000)))
    return {k: shards.fingerprint(m, probe) for k, m in env.ss.models.items()}


def test_20_requests_over_3_shards_cause_exactly_3_retrains(env, trains):
    reqs = twenty_requests(env)
    assert len(reqs) == 20 and {k for _, k, _ in reqs} == set(SHARDS)
    before = fps(env)
    q = new_queue(env)
    for rid, _, pid in reqs:
        q.submit([pid], rid)
    assert q.pending() == 20 and trains == []          # submit never retrains

    receipts = q.flush()

    assert sorted(trains) == sorted(SHARDS) and len(trains) == 3
    assert q.pending() == 0 and len(receipts) == 20
    assert [r["request_id"] for r in receipts] == [rid for rid, _, _ in reqs]   # submission order
    after = fps(env)
    for k in range(8):
        assert (after[k] != before[k]) == (k in SHARDS)

    shard_of_req = {rid: k for rid, k, _ in reqs}
    for r in receipts:
        assert R.verify_receipt(r, KEY)
        assert r["shard_ids"] == [shard_of_req[r["request_id"]]]
        assert r["persons_erased"] == 1 and r["rows_removed"] == 1 and r["batch_size"] == 20
        k = str(r["shard_ids"][0])
        assert r["model_hash_before"][k] == before[int(k)] and r["model_hash_after"][k] == after[int(k)]
        assert R.verify_receipt_log_linkage(r, env.log_path, verify_chain=False)[0]
    # shared batch stats
    for field in ("full_retrain_seconds", "shard_retrain_seconds", "speedup",
                  "test_auroc_before", "test_auroc_after", "guard", "timestamp_utc"):
        assert len({repr(r[field]) for r in receipts}) == 1, field
    assert receipts[0]["shard_retrain_seconds"] > 0
    assert [r["log_seq"] for r in receipts] == list(range(1, 21))
    assert R.verify_log(env.log_path) == (True, None)

    cur = Q.load_train_df(env.data_dir)
    assert not ({p for _, _, p in reqs} & set(cur.person_id))
    assert len(cur) == len(env.train) - 20


def test_queue_result_equals_sequential_erases_exactness(env, trains, tmp_path, monkeypatch):
    reqs = twenty_requests(env)[:9]
    q = new_queue(env, measure_full_retrain=False)
    for rid, _, pid in reqs:
        q.submit([pid], rid)
    q.flush()
    queued = fps(env)

    env2 = F.make_env(tmp_path / "seq")
    train = env2.train
    for rid, _, pid in reqs:
        Q.erase([pid], rid, env2.ss, train, data_dir=env2.data_dir, log_path=env2.log_path,
                measure_full_retrain=False)
        train = Q.load_train_df(env2.data_dir)
    assert fps(env2) == queued


def test_concurrent_submits_then_one_flush(env, trains):
    reqs = twenty_requests(env)
    q = new_queue(env, measure_full_retrain=False)
    chunks = [reqs[i::4] for i in range(4)]

    def worker(chunk):
        for rid, _, pid in chunk:
            q.submit([pid], rid)

    ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert q.pending() == 20
    receipts = q.flush()
    assert len(receipts) == 20 and sorted(trains) == sorted(SHARDS)
    assert R.verify_log(env.log_path) == (True, None)


def test_requests_spanning_shards_and_overlapping_people(env, trains):
    a = F.persons_in_shard(env, 1, 2, member=True)
    b = F.persons_in_shard(env, 4, 2, member=True)
    q = new_queue(env, measure_full_retrain=False)
    q.submit(a + b, "REQ-A")          # spans two shards
    q.submit([a[0]], "REQ-B")         # same person again: already handled by REQ-A
    q.submit(b[1:], "REQ-C")          # also already handled by REQ-A
    ra, rb, rc = q.flush()
    assert sorted(trains) == [1, 4]
    assert ra["shard_ids"] == [1, 4] and ra["persons_erased"] == 4 and ra["rows_removed"] == 4
    for r in (rb, rc):
        assert r["persons_erased"] == 0 and r["shard_ids"] == [] and r["persons_not_found"] == 1
        assert R.verify_receipt(r, KEY)


def test_second_flush_uses_remaining_data(env, trains):
    a, b = F.persons_in_shard(env, 1, 2, member=True)
    q = new_queue(env, measure_full_retrain=False)
    q.submit([a], "REQ-0001")
    q.flush()
    q.submit([b], "REQ-0002")
    (r2,) = q.flush()
    probe = env.test[FEATURES].head(int(THRESHOLDS.get("probe_rows", 1000)))
    remaining = env.train[(env.shard_col == 1) & ~env.train.person_id.isin([a, b])]
    assert r2["model_hash_after"]["1"] == shards.fingerprint(Q._REAL_TRAIN_SHARD(remaining, 1), probe)
    assert trains == [1, 1]
    assert q.flush() == []            # empty flush is a no-op


def test_failed_flush_requeues_requests_and_changes_nothing(env, monkeypatch):
    ids = F.persons_in_shard(env, 1, 2, member=True)
    q = new_queue(env, measure_full_retrain=False)
    q.submit([ids[0]], "REQ-0001")
    q.submit([ids[1]], "REQ-0002")
    before = fps(env)
    orig = shards.train_shard
    monkeypatch.setattr(shards, "train_shard", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        q.flush()
    assert q.pending() == 2 and fps(env) == before
    monkeypatch.setattr(shards, "train_shard", orig)
    assert [r["request_id"] for r in q.flush()] == ["REQ-0001", "REQ-0002"]


def test_submit_validation(env):
    q = new_queue(env)
    q.submit([1], "REQ-0001")
    with pytest.raises(ValueError):
        q.submit([2], "REQ-0001")        # duplicate pending id
    with pytest.raises(ValueError):
        q.submit([2], "bad id")
    with pytest.raises(ValueError):
        q.submit(["x"], "REQ-0002")
    assert q.pending() == 1


def test_queue_loads_current_training_set_when_none_given(env, trains):
    a, b = F.persons_in_shard(env, 1, 2, member=True)
    Q.erase([a], "REQ-0001", env.ss, env.train, data_dir=env.data_dir, log_path=env.log_path,
            measure_full_retrain=False)
    q = Q.EraseQueue(env.ss, data_dir=env.data_dir, log_path=env.log_path, measure_full_retrain=False)
    assert a not in set(q.train_df.person_id)
    q.submit([a], "REQ-0002")             # already erased -> nothing to do
    (r,) = q.flush()
    assert r["persons_erased"] == 0
