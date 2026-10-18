"""FR-7: hash-chained log, circular-hash fix, receipt<->log linkage."""
import json
import multiprocessing as mp
import os
import threading

import pytest

from receipt_fixtures import KEY, make_pre_receipt
from src.report import receipt as R

H = "e" * 64


def entry(i, shards=(1,)):
    return {"request_id": f"REQ-{i:04d}", "timestamp_utc": "2026-10-17T16:10:12Z",
            "shard_ids": list(shards), "receipt_sha256": H}


def build_log(path, n=5):
    return [R.append_log(entry(i, (i % 8,)), str(path)) for i in range(1, n + 1)]


def read_lines(path):
    return path.read_bytes().split(b"\n")[:-1]


def write_lines(path, lines):
    path.write_bytes(b"\n".join(lines) + b"\n")


def test_clean_chain_true_and_format(tmp_path):
    p = tmp_path / "log.jsonl"
    hashes = build_log(p, 5)
    assert R.verify_log(str(p)) == (True, None)
    entries = [json.loads(x) for x in read_lines(p)]
    assert [e["seq"] for e in entries] == [1, 2, 3, 4, 5]
    assert set(entries[0]) == {"seq", "request_id", "timestamp_utc", "shard_ids",
                               "receipt_sha256", "prev_hash", "entry_hash"}
    assert entries[0]["prev_hash"] == "0" * 64
    for a, b in zip(entries, entries[1:]):
        assert b["prev_hash"] == a["entry_hash"]
    assert hashes == [e["entry_hash"] for e in entries]
    import hashlib
    e = dict(entries[2])
    stored = e.pop("entry_hash")
    raw = json.dumps(e, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == stored


def test_missing_and_empty_log_are_valid(tmp_path):
    assert R.verify_log(str(tmp_path / "nope.jsonl")) == (True, None)
    (tmp_path / "e.jsonl").write_bytes(b"")
    assert R.verify_log(str(tmp_path / "e.jsonl")) == (True, None)


def test_edit_middle_entry_returns_its_seq(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 5)
    lines = read_lines(p)
    e = json.loads(lines[2])
    e["shard_ids"] = [7]
    lines[2] = json.dumps(e, sort_keys=True, separators=(",", ":")).encode()
    write_lines(p, lines)
    assert R.verify_log(str(p)) == (False, 3)


def test_edit_with_recomputed_own_hash_breaks_at_next_entry(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 5)
    lines = read_lines(p)
    e = json.loads(lines[2])
    e["request_id"] = "REQ-FAKE"
    e["entry_hash"] = R.entry_hash_of(e)
    lines[2] = R.canonical_bytes(e)
    write_lines(p, lines)
    assert R.verify_log(str(p)) == (False, 4)


def test_delete_middle_entry_returns_that_seq(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 5)
    lines = read_lines(p)
    del lines[2]
    write_lines(p, lines)
    assert R.verify_log(str(p)) == (False, 3)


def test_first_entry_edit_and_garbage_line(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 3)
    lines = read_lines(p)
    e = json.loads(lines[0])
    e["receipt_sha256"] = "f" * 64
    lines[0] = R.canonical_bytes(e)
    write_lines(p, lines)
    assert R.verify_log(str(p)) == (False, 1)
    build = tmp_path / "log2.jsonl"
    build_log(build, 3)
    ls = read_lines(build)
    ls[1] = b"not json"
    write_lines(build, ls)
    assert R.verify_log(str(build)) == (False, 2)


def test_swapped_entries_detected(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 4)
    lines = read_lines(p)
    lines[1], lines[2] = lines[2], lines[1]
    write_lines(p, lines)
    assert R.verify_log(str(p)) == (False, 2)


def test_append_refuses_to_extend_corrupt_log(tmp_path):
    p = tmp_path / "log.jsonl"
    build_log(p, 2)
    with open(p, "ab") as fh:
        fh.write(b'{"seq":3')  # torn write, no newline
    with pytest.raises(R.LogCorruptError):
        R.append_log(entry(3), str(p))


def test_log_input_rejects_patient_fields_and_ignores_caller_chain_fields(tmp_path):
    p = str(tmp_path / "log.jsonl")
    with pytest.raises(ValueError):
        R.append_log({**entry(1), "person_ids": [1000001]}, p)
    with pytest.raises(ValueError):
        R.append_log({**entry(1), "shard_ids": [1000001]}, p)
    with pytest.raises(ValueError):
        R.append_log({**entry(1), "receipt_sha256": "nothex"}, p)
    h = R.append_log({**entry(1), "seq": 99, "prev_hash": "1" * 64, "entry_hash": "2" * 64}, p)
    stored = json.loads(read_lines(tmp_path / "log.jsonl")[0])
    assert stored["seq"] == 1 and stored["prev_hash"] == "0" * 64 and stored["entry_hash"] == h


def _proc_worker(path, start, n):
    for i in range(n):
        R.append_log(entry(start + i), path)


@pytest.mark.linux_only
def test_concurrent_appends_threads_and_processes(tmp_path):
    p = str(tmp_path / "log.jsonl")
    threads = [threading.Thread(target=_proc_worker, args=(p, t * 100, 10)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    procs = [ctx.Process(target=_proc_worker, args=(p, 1000 + k * 100, 10)) for k in range(4)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        assert pr.exitcode == 0
    entries = [json.loads(x) for x in read_lines(tmp_path / "log.jsonl")]
    assert [e["seq"] for e in entries] == list(range(1, 121))
    assert R.verify_log(p) == (True, None)


# ---- circular-hash fix + linkage ------------------------------------------
def test_finalize_receipt_links_receipt_and_log(tmp_path):
    p = str(tmp_path / "log.jsonl")
    R.append_log(entry(0), p)  # so the receipt is not seq 1
    pre = make_pre_receipt()
    final = R.finalize_receipt(pre, KEY, p)
    assert final["log_seq"] == 2 and R.verify_receipt(final, KEY)
    stored = R.read_log_entry(2, p)
    assert stored["entry_hash"] == final["log_entry_hash"]
    assert stored["receipt_sha256"] == R.receipt_sha256(pre) == R.receipt_sha256(final)
    assert R.verify_receipt_log_linkage(final, p) == (True, "linked")
    assert R.verify_log(p) == (True, None)


def test_pre_log_hash_excludes_exactly_log_fields_and_hmac():
    pre = make_pre_receipt()
    final = dict(pre, log_seq=5, log_entry_hash="a" * 64)
    final["hmac"] = R.sign(final, KEY)
    assert R.receipt_sha256(pre) == R.receipt_sha256(final)
    assert R.receipt_sha256(dict(pre, rows_removed=7)) != R.receipt_sha256(pre)


def test_linkage_detects_receipt_edit_even_if_resigned(tmp_path):
    p = str(tmp_path / "log.jsonl")
    final = R.finalize_receipt(make_pre_receipt(), KEY, p)
    forged = dict(final, rows_removed=0)
    forged["hmac"] = R.sign(forged, KEY)  # attacker holding the key still cannot fix the log
    assert R.verify_receipt(forged, KEY)
    ok, why = R.verify_receipt_log_linkage(forged, p)
    assert not ok and "receipt content" in why


def test_linkage_detects_log_tamper_and_truncation(tmp_path):
    p = tmp_path / "log.jsonl"
    finals = [R.finalize_receipt(make_pre_receipt(request_id=f"REQ-{i:04d}"), KEY, str(p))
              for i in range(1, 4)]
    assert all(R.verify_receipt_log_linkage(f, str(p))[0] for f in finals)
    lines = read_lines(p)
    # truncate the tail: receipt 3 now points past the end of the log
    write_lines(p, lines[:2])
    ok, why = R.verify_receipt_log_linkage(finals[2], str(p))
    assert not ok and "no entry" in why
    # edit a middle entry: chain check fails for every receipt
    write_lines(p, lines)
    e = json.loads(lines[1])
    e["shard_ids"] = [5]
    lines[1] = R.canonical_bytes(e)
    write_lines(p, lines)
    ok, why = R.verify_receipt_log_linkage(finals[0], str(p))
    assert not ok and "seq 2" in why
    assert R.verify_receipt_log_linkage(finals[0], str(p), verify_chain=False)[0] is True


def test_finalize_rejects_invalid_receipt(tmp_path):
    p = str(tmp_path / "log.jsonl")
    bad = dict(make_pre_receipt(), person_id=1000001)
    with pytest.raises(ValueError):
        R.finalize_receipt(bad, KEY, p)
    assert not (tmp_path / "log.jsonl").exists()


# ---- scripts/tamper_demo.py -------------------------------------------------
def test_tamper_demo_detects_everything(monkeypatch, tmp_path):
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "tamper_demo.py"
    spec = importlib.util.spec_from_file_location("tamper_demo", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv(R.HMAC_ENV, "k-for-test")
    res = mod.run_demo(str(tmp_path / "demo"))
    assert res["all_detected"] is True
    names = {s["name"]: s for s in res["steps"]}
    assert names["receipt with ONE flipped byte"]["valid"] is False
    assert names["log with a middle entry edited"]["valid"] is False
    assert names["log with a middle entry deleted"]["valid"] is False
    assert names["clean receipt verifies"]["valid"] is True

    # fresh process with NO key set: the demo supplies its own throwaway key and exits 0
    env = {k: v for k, v in os.environ.items() if k != R.HMAC_ENV}
    p = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0 and "All tampering detected." in p.stdout
