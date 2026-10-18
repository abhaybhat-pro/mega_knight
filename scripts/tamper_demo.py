#!/usr/bin/env python3
"""Tamper demo: shows that a modified receipt and a modified log are both caught.

Uses ILLUSTRATIVE PLACEHOLDER numbers (not measurements) in a throwaway temp
directory, so it never touches the real logs/ or models/. Sets a throwaway
FORGETME_HMAC_KEY only if none is set (demo value, never a real key).

    python scripts/tamper_demo.py            # prints the story, exit 0 if all tampering detected
    python scripts/tamper_demo.py --out DIR  # also keep the demo receipt/log files in DIR

The UI can call ``run_demo()`` and render the returned dict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report import receipt as R  # noqa: E402

DEMO_KEY = "forgetme-demo-throwaway-key"


def _view(auc: float) -> dict:
    return {"attack_auc": auc, "auc_ci95": [round(auc - 0.05, 3), round(min(1.0, auc + 0.03), 3)],
            "tpr_at_1pct_fpr": 0.5, "canary_percentile_median": 0.9}


def _demo_receipt(i: int) -> dict:
    return R.build_receipt(
        request_id=f"DEMO-{i:04d}", persons_erased=2, shard_ids=[i % 8], rows_removed=2,
        model_hash_before={str(i % 8): "a" * 64}, model_hash_after={str(i % 8): "b" * 64},
        full_retrain_seconds=8.0, shard_retrain_seconds=1.0, speedup=8.0,
        test_auroc_before=0.8, test_auroc_after=0.8,
        audit_before={"ensemble": _view(0.7), "shard": _view(0.95)},
        audit_after={"ensemble": _view(0.5), "shard": _view(0.5)},
        note="Tamper demo with placeholder values",
    )


def _flip_byte(raw: bytes, idx: int) -> bytes:
    b = bytearray(raw)
    b[idx] ^= 0x01
    return bytes(b)


def run_demo(workdir: str | None = None) -> dict:
    """Run the full demo and return what was detected. Raises nothing on detection failures."""
    if not os.environ.get(R.HMAC_ENV):
        os.environ[R.HMAC_ENV] = DEMO_KEY
    key = R.get_hmac_key()
    ctx = None
    if workdir is None:
        ctx = tempfile.TemporaryDirectory(prefix="forgetme_tamper_")
        workdir = ctx.name
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "erasure_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    try:
        receipts = [R.finalize_receipt(_demo_receipt(i), key, str(log_path)) for i in range(1, 5)]
        rpath = work / "receipt_DEMO-0002.json"
        rpath.write_bytes(R.receipt_bytes(receipts[1]))
        out: dict = {"receipt_path": str(rpath), "log_path": str(log_path), "steps": []}

        def step(name: str, expect_ok: bool, got_ok: bool, detail: str = "") -> None:
            out["steps"].append({"name": name, "expected_valid": expect_ok, "valid": got_ok,
                                 "correct": expect_ok == got_ok, "detail": detail})

        # 0. clean
        step("clean receipt verifies", True, R.verify_receipt_bytes(rpath.read_bytes(), key))
        ok, bad = R.verify_log(str(log_path))
        step("clean log verifies", True, ok, f"first_bad_seq={bad}")
        step("receipt is linked to its log entry", True,
             R.verify_receipt_log_linkage(receipts[1], str(log_path))[0])

        # 1. flip ONE byte of the receipt file
        raw = rpath.read_bytes()
        idx = raw.index(b'"rows_removed":') + len(b'"rows_removed":')   # flips the digit 2 -> 3
        tampered = _flip_byte(raw, idx)
        step("receipt with ONE flipped byte", False, R.verify_receipt_bytes(tampered, key),
             f"byte {idx}: {raw[idx:idx + 1].decode()} -> {tampered[idx:idx + 1].decode()}")

        # 2. edit a value and keep the old signature
        edited = dict(receipts[1], rows_removed=0)
        step("receipt value edited (signature kept)", False, R.verify_receipt(edited, key))

        # 3. edit a value AND re-sign (attacker with the key) -> the log still disagrees
        forged = R.sign_receipt(edited, key)
        ok, why = R.verify_receipt_log_linkage(forged, str(log_path))
        step("receipt edited and re-signed (log cross-check)", False, ok, why)

        # 4. edit a middle log entry
        lines = log_path.read_bytes().split(b"\n")[:-1]
        entry = json.loads(lines[1])
        entry["shard_ids"] = [7]
        edited_lines = list(lines)
        edited_lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        log_path.write_bytes(b"\n".join(edited_lines) + b"\n")
        ok, bad = R.verify_log(str(log_path))
        step("log with a middle entry edited", False, ok, f"chain breaks at seq {bad}")

        # 5. delete a middle log entry
        del_lines = lines[:1] + lines[2:]
        log_path.write_bytes(b"\n".join(del_lines) + b"\n")
        ok, bad = R.verify_log(str(log_path))
        step("log with a middle entry deleted", False, ok, f"chain breaks at seq {bad}")

        log_path.write_bytes(b"\n".join(lines) + b"\n")   # restore
        ok, _ = R.verify_log(str(log_path))
        step("log restored verifies again", True, ok)

        out["all_detected"] = all(s["correct"] for s in out["steps"])
        if ctx is not None:  # keep nothing behind, but report paths as they were
            out["receipt_path"] = out["log_path"] = "(temporary directory removed; use --out to keep files)"
        return out
    finally:
        if ctx is not None:
            ctx.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="directory to keep the demo receipt and log in")
    args = ap.parse_args()
    res = run_demo(args.out)
    print("ForgetMe tamper demo (placeholder values, throwaway key)\n")
    for s in res["steps"]:
        mark = "OK " if s["correct"] else "BAD"
        verdict = "valid" if s["valid"] else "REJECTED"
        extra = f"  [{s['detail']}]" if s["detail"] else ""
        print(f"  [{mark}] {s['name']}: {verdict}{extra}")
    print()
    print("All tampering detected." if res["all_detected"] else "FAILURE: some tampering was NOT detected.")
    if args.out:
        print(f"receipt: {res['receipt_path']}\nlog:     {res['log_path']}")
    return 0 if res["all_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
