#!/usr/bin/env python3
"""V6: integrity checks on the demo's REAL receipt and log, plus byte-flip tamper tests on COPIES.

Needs FORGETME_HMAC_KEY set to the key the demo ran with (run `python scripts/run_demo.py` with a key you export).
Exit 0 only if the real artifacts verify and both tampered copies are detected. Nothing real is modified.
"""
import json, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report import receipt as R  # noqa: E402

key = os.environ.get("FORGETME_HMAC_KEY", "").encode()
if not key:
    sys.exit("FORGETME_HMAC_KEY is not set")
rec_path, log_path = ROOT / "receipts" / "REQ-0001.json", ROOT / "logs" / "erasure_log.jsonl"
raw = rec_path.read_bytes()
rec = json.loads(raw)
res = {}
res["verify_receipt (real)"] = R.verify_receipt(rec, key)
res["verify_log (real)"] = R.verify_log(str(log_path))[0]
res["receipt-to-log linkage (real)"] = R.verify_receipt_log_linkage(rec, str(log_path))[0]

tmp = Path(tempfile.mkdtemp(prefix="v6_"))
# (a) flip one byte of a digit inside the receipt copy
i = next(k for k, b in enumerate(raw) if k > len(raw) // 2 and chr(b).isdigit() and chr(raw[k - 1]).isdigit() and chr(raw[k + 1]).isdigit())
bad = bytearray(raw); bad[i] = ord("0") + (raw[i] - ord("0") + 1) % 10
try:
    flipped = json.loads(bytes(bad))
    res["tampered receipt copy detected (one byte flipped, still valid JSON)"] = not R.verify_receipt(flipped, key)
except ValueError:
    res["tampered receipt copy detected (one byte flipped, no longer parses)"] = True
# (b) COPY of the log, extended with two entries, then one byte flipped in the middle entry
lp = tmp / "log.jsonl"; shutil.copy(log_path, lp)
first = json.loads(log_path.read_text().splitlines()[0])
for n in (2, 3):
    R.append_log_entry({"request_id": f"REQ-000{n}", "timestamp_utc": first["timestamp_utc"],
                        "shard_ids": first["shard_ids"], "receipt_sha256": first["receipt_sha256"]}, str(lp))
assert R.verify_log(str(lp))[0], "extended copy should verify before tampering"
lines = lp.read_bytes().split(b"\n")
mid = bytearray(lines[1]); j = mid.index(b"REQ-0002") + 7
mid[j] = ord("9"); lines[1] = bytes(mid)
lp.write_bytes(b"\n".join(lines))
ok, bad_seq = R.verify_log(str(lp))
res[f"tampered middle log entry detected (first bad seq {bad_seq})"] = not ok
for k, v in res.items():
    print(("PASS  " if v else "FAIL  ") + k)
sys.exit(0 if all(res.values()) else 1)
