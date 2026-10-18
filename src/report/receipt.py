"""Signed erasure receipts and the hash-chained erasure log (FR-6, FR-7).


Receipt
-------
* ``build_receipt(**fields)`` validates against the 8.4 schema and REJECTS
  anything patient-level: top-level keys are whitelisted, nested metric blocks
  may only hold numbers / short hex hashes / short labels, lists are capped,
  ``shard_ids`` must be small integers, and a deny-list of patient field names
  is enforced at every depth.
* ``sign`` = HMAC-SHA256 over the receipt with the ``hmac`` field removed,
  serialised with sorted keys, compact separators ``(",", ":")``, UTF-8.
  The key comes from ``FORGETME_HMAC_KEY`` only (no default).
* ``verify_receipt`` uses ``hmac.compare_digest``.

Two verification levels
-----------------------
``verify_receipt(dict)`` protects every *value* in the receipt. A byte flip in
a receipt file that merely re-spells the same value (e.g. ``1e-05`` -> ``1E-05``)
parses to the same dict, so it cannot be seen at dict level.
``verify_receipt_bytes(raw)`` closes that gap: it additionally requires the raw
bytes to be byte-identical to the canonical serialisation, so *any* single byte
change in a canonical receipt file fails.

Circular-hash fix (spec gap 3)
------------------------------
The log entry needs the receipt's hash, but the receipt wants the log's
``seq`` / ``entry_hash``. Resolution:

1. Build the *pre-log receipt* (no ``log_seq``, ``log_entry_hash``, ``hmac``).
2. ``receipt_sha256 = sha256(canonical(pre-log receipt))``.
3. Append the log entry carrying that ``receipt_sha256``.
4. Add ``log_seq`` and ``log_entry_hash`` to the receipt, then sign with HMAC.

``finalize_receipt`` does all four steps. ``verify_receipt_log_linkage``
re-derives the pre-log receipt from a final receipt and checks it against the
log entry (and, optionally, the whole chain).

Known limitation: deleting the *last* entries of a log is not detectable from
the log alone (there is nothing after them to break). A receipt that carries a
``log_seq`` beyond the log's length is caught by the linkage helper.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

try:  # POSIX only; the target VM is Linux (s390x).
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from src.contracts import LOG_PATH

HMAC_ENV = "FORGETME_HMAC_KEY"
SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
DEFAULT_LOG_PATH = LOG_PATH

# Fields removed to form the pre-log receipt (circular-hash fix).
PRE_LOG_EXCLUDED = ("log_seq", "log_entry_hash", "hmac")
LOG_ENTRY_FIELDS = (
    "seq", "request_id", "timestamp_utc", "shard_ids",
    "receipt_sha256", "prev_hash", "entry_hash",
)
_LOG_INPUT_FIELDS = {"request_id", "timestamp_utc", "shard_ids", "receipt_sha256"}

# --------------------------------------------------------------------------- #
# Schema / whitelist
# --------------------------------------------------------------------------- #
REQUIRED_KEYS = (
    "request_id", "persons_erased", "shard_ids", "rows_removed",
    "model_hash_before", "model_hash_after",
    "full_retrain_seconds", "shard_retrain_seconds", "speedup",
    "test_auroc_before", "test_auroc_after",
    "audit_before", "audit_after", "guard",
)
ALLOWED_KEYS = frozenset(
    REQUIRED_KEYS
    + ("schema_version", "timestamp_utc", "log_seq", "log_entry_hash", "hmac",
       # documented extensions (counts / metrics / fixed notes only):
       "shard_detail", "note", "persons_not_found", "batch_size",
       # SC-12: serial full-retrain timing. speedup stays parallel / shard retrain.
       "full_retrain_serial_seconds", "speedup_vs_serial")
)
GUARD_KEYS = frozenset(
    ("worst_subgroup", "max_subgroup_auroc_drop", "drift_flags",
     "held_decisions", "flagged_subgroups")
)

# Names that must never appear as a key at any depth.
PATIENT_FIELD_DENYLIST = frozenset(
    ("person_id", "person_ids", "patient_id", "patient_ids", "erased_ids", "ids",
     "id_list", "name", "patient", "patients", "mrn", "label", "labels",
     "canary_type", "needs_referral", "age", "sex", "bmi", "sbp", "dbp",
     "hba1c", "smoker", "sym_fatigue", "sym_vision", "sym_thirst",
     "sym_chest_pain", "village_code", "prior_visits", "features", "rows", "data")
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQID = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_KEY = re.compile(r"^[a-z0-9_]{1,48}$")
_LABEL = re.compile(r"^[A-Za-z0-9_.:+\-]{1,40}$")
_NOTE = re.compile(r"^[A-Za-z0-9 .,;:()_\-/%+']{0,200}$")
_LONG_DIGITS = re.compile(r"\d{5,}")  # person IDs are 6-7 digits; keep them out of notes
_MAX_LIST = 64
_MAX_SHARD_ID = 1023


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_request_id(request_id: Any) -> str:
    """Raise ValueError unless request_id is receipt/log-safe. Used to fail fast
    BEFORE an erase touches any model."""
    if not (isinstance(request_id, str) and _REQID.fullmatch(request_id)):
        raise ValueError("request_id must match [A-Za-z0-9_.:-]{1,64}")
    return request_id


# --------------------------------------------------------------------------- #
# Canonical serialisation / hashing
# --------------------------------------------------------------------------- #
def _native(o: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and non-str keys to plain JSON types."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return [_native(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    return o


def canonical_bytes(obj: Any) -> bytes:
    """Sorted keys, compact separators, UTF-8, no NaN/Infinity."""
    return json.dumps(
        _native(obj), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pre_log_receipt(receipt: dict) -> dict:
    """The receipt as it was when ``receipt_sha256`` was computed."""
    return {k: v for k, v in receipt.items() if k not in PRE_LOG_EXCLUDED}


def receipt_sha256(receipt: dict) -> str:
    """sha256 over the canonical pre-log receipt (no log_seq/log_entry_hash/hmac)."""
    return sha256_hex(canonical_bytes(pre_log_receipt(receipt)))


# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #
def get_hmac_key() -> bytes:
    """Key bytes from FORGETME_HMAC_KEY. There is deliberately no default."""
    val = os.environ.get(HMAC_ENV)
    if not val:
        raise RuntimeError(
            f"{HMAC_ENV} is not set. Export a secret key before signing or "
            f"verifying receipts (e.g. `export {HMAC_ENV}=...`); there is no default."
        )
    return val.encode("utf-8")


def _require_key(key: Any) -> bytes:
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise ValueError("HMAC key must be non-empty bytes")
    return bytes(key)


# --------------------------------------------------------------------------- #
# Validation (patient-data rejection)
# --------------------------------------------------------------------------- #
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_key(k: Any, path: str) -> None:
    if not isinstance(k, str) or not _KEY.fullmatch(k):
        raise ValueError(f"{path}: key {k!r} is not an allowed metric key")
    if k.lower() in PATIENT_FIELD_DENYLIST:
        raise ValueError(f"{path}: patient-level field name {k!r} is not allowed in a receipt")


def _validate_metrics(obj: Any, path: str, *, allow_hex: bool = False, depth: int = 0) -> None:
    """Metric blocks may only hold numbers/None/bools (and hex hashes if allowed)."""
    if depth > 6:
        raise ValueError(f"{path}: nested too deep")
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite number")
        return
    if isinstance(obj, str):
        if allow_hex and _HEX64.fullmatch(obj):
            return
        raise ValueError(f"{path}: free-text strings are not allowed here")
    if isinstance(obj, list):
        if len(obj) > _MAX_LIST:
            raise ValueError(f"{path}: list longer than {_MAX_LIST} (possible ID list)")
        for i, v in enumerate(obj):
            _validate_metrics(v, f"{path}[{i}]", allow_hex=allow_hex, depth=depth + 1)
        return
    if isinstance(obj, dict):
        if len(obj) > _MAX_LIST:
            raise ValueError(f"{path}: too many keys")
        for k, v in obj.items():
            _check_key(k, path)
            _validate_metrics(v, f"{path}.{k}", allow_hex=allow_hex, depth=depth + 1)
        return
    raise ValueError(f"{path}: unsupported type {type(obj).__name__}")


def _check_audit(name: str, v: Any) -> None:
    if v is None:
        return
    if not isinstance(v, dict) or not set(v) <= {"ensemble", "shard"}:
        raise ValueError(f"{name}: must be null or a dict with only 'ensemble'/'shard'")
    for view, block in v.items():
        if block is not None:
            if not isinstance(block, dict):
                raise ValueError(f"{name}.{view}: must be a dict or null")
            _validate_metrics(block, f"{name}.{view}")


def _check_label(x: Any, path: str) -> None:
    if not (isinstance(x, str) and _LABEL.fullmatch(x)):
        raise ValueError(f"{path}: must be a short label (letters, digits, _ . : + -)")


def _check_guard(g: Any) -> None:
    if not isinstance(g, dict) or not set(g) <= GUARD_KEYS:
        raise ValueError(f"guard: keys must be within {sorted(GUARD_KEYS)}")
    ws = g.get("worst_subgroup")
    if ws is not None:
        _check_label(ws, "guard.worst_subgroup")
    d = g.get("max_subgroup_auroc_drop")
    if d is not None and not _is_num(d):
        raise ValueError("guard.max_subgroup_auroc_drop must be a number or null")
    df = g.get("drift_flags")
    if df is not None:
        if _is_int(df):
            pass
        elif isinstance(df, list) and len(df) <= _MAX_LIST:
            for i, x in enumerate(df):
                _check_label(x, f"guard.drift_flags[{i}]")
        else:
            raise ValueError("guard.drift_flags must be null, a count, or a list of labels")
    hd = g.get("held_decisions")
    if hd is not None and not (_is_int(hd) and hd >= 0):
        raise ValueError("guard.held_decisions must be null or a count (never patient IDs)")
    fs = g.get("flagged_subgroups")
    if fs is not None:
        if not isinstance(fs, dict) or len(fs) > _MAX_LIST:
            raise ValueError("guard.flagged_subgroups must be null or {label: auroc_drop}")
        for k, v in fs.items():
            _check_label(k, "guard.flagged_subgroups key")
            if not _is_num(v):
                raise ValueError("guard.flagged_subgroups values must be numbers")


def validate_receipt_fields(r: dict) -> None:
    """Raise ValueError unless ``r`` fits the 8.4 schema and holds no patient data."""
    if not isinstance(r, dict):
        raise ValueError("receipt must be a dict")
    unknown = set(r) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"fields not allowed in a receipt: {sorted(unknown)}")
    missing = [k for k in REQUIRED_KEYS if k not in r]
    if missing:
        raise ValueError(f"missing required receipt fields: {missing}")
    for k in r:  # deny-list belt and braces (top level is already whitelisted)
        if k.lower() in PATIENT_FIELD_DENYLIST:
            raise ValueError(f"patient-level field {k!r} not allowed")

    if r.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION or isinstance(r.get("schema_version"), bool):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not (isinstance(r["request_id"], str) and _REQID.fullmatch(r["request_id"])):
        raise ValueError("request_id must match [A-Za-z0-9_.:-]{1,64}")
    ts = r.get("timestamp_utc")
    if ts is not None and not (isinstance(ts, str) and _TS.fullmatch(ts)):
        raise ValueError("timestamp_utc must look like 2026-10-17T16:10:12Z")

    for k in ("persons_erased", "rows_removed", "persons_not_found", "batch_size"):
        if k in r and not (_is_int(r[k]) and r[k] >= 0):
            raise ValueError(f"{k} must be a non-negative integer")

    sids = r["shard_ids"]
    if not isinstance(sids, list) or len(sids) > _MAX_LIST:
        raise ValueError("shard_ids must be a short list of shard numbers")
    if not all(_is_int(s) and 0 <= s <= _MAX_SHARD_ID for s in sids) or len(set(sids)) != len(sids):
        raise ValueError(f"shard_ids must be unique integers in 0..{_MAX_SHARD_ID} (never person IDs)")

    for k in ("model_hash_before", "model_hash_after"):
        h = r[k]
        if not isinstance(h, dict) or set(h) != {str(s) for s in sids}:
            raise ValueError(f"{k} must be a dict keyed by the shard ids {sorted(sids)}")
        for hk, hv in h.items():
            if not (isinstance(hv, str) and _HEX64.fullmatch(hv)):
                raise ValueError(f"{k}[{hk}] must be a 64-char lowercase hex digest")

    for k in ("full_retrain_seconds", "shard_retrain_seconds", "speedup"):
        if not (_is_num(r[k]) and r[k] >= 0):
            raise ValueError(f"{k} must be a finite number >= 0")
    for k in ("full_retrain_serial_seconds", "speedup_vs_serial"):
        if k in r and r[k] is not None and not (_is_num(r[k]) and r[k] >= 0):
            raise ValueError(f"{k} must be a finite number >= 0")
    for k in ("test_auroc_before", "test_auroc_after"):
        if not (_is_num(r[k]) and 0.0 <= r[k] <= 1.0):
            raise ValueError(f"{k} must be a number in [0, 1]")

    _check_audit("audit_before", r["audit_before"])
    _check_audit("audit_after", r["audit_after"])
    _check_guard(r["guard"])

    if "shard_detail" in r and r["shard_detail"] is not None:
        sd = r["shard_detail"]
        if not isinstance(sd, dict) or not all(isinstance(k, str) and k.isdigit() for k in sd):
            raise ValueError("shard_detail must be a dict keyed by shard number")
        _validate_metrics(sd, "shard_detail", allow_hex=True)
    if "note" in r and r["note"] is not None:
        if not (isinstance(r["note"], str) and _NOTE.fullmatch(r["note"])) or _LONG_DIGITS.search(r["note"]):
            raise ValueError("note must be short plain text (<=200 chars, no long digit runs / IDs)")

    if "log_seq" in r and not (_is_int(r["log_seq"]) and r["log_seq"] >= 1):
        raise ValueError("log_seq must be an integer >= 1")
    for k in ("log_entry_hash", "hmac"):
        if k in r and not (isinstance(r[k], str) and _HEX64.fullmatch(r[k])):
            raise ValueError(f"{k} must be a 64-char lowercase hex digest")


def build_receipt(**fields: Any) -> dict:
    """Validate and return a receipt dict (unsigned, not yet logged).

    ``schema_version`` and ``timestamp_utc`` default sensibly; ``guard``
    defaults to all-null (Guard not built). Everything else in
    ``REQUIRED_KEYS`` must be supplied. Raises ValueError on unknown fields,
    patient-level data, or schema violations.
    """
    r = _native(dict(fields))
    r.setdefault("schema_version", SCHEMA_VERSION)
    r.setdefault("timestamp_utc", utc_now())
    r.setdefault("guard", {"worst_subgroup": None, "max_subgroup_auroc_drop": None,
                           "drift_flags": None, "held_decisions": None})
    validate_receipt_fields(r)
    return copy.deepcopy(r)


# --------------------------------------------------------------------------- #
# HMAC sign / verify
# --------------------------------------------------------------------------- #
def sign(receipt: dict, key: bytes) -> str:
    """HMAC-SHA256 hex over the canonical receipt with 'hmac' removed."""
    k = _require_key(key)
    body = {x: v for x, v in receipt.items() if x != "hmac"}
    return hmac.new(k, canonical_bytes(body), hashlib.sha256).hexdigest()


def verify_receipt(receipt: dict, key: bytes) -> bool:
    """Constant-time check of the receipt's ``hmac`` field. Never raises on bad data."""
    k = _require_key(key)
    try:
        stored = receipt.get("hmac")
        if not isinstance(stored, str):
            return False
        expected = sign(receipt, k)
        return hmac.compare_digest(expected.encode("ascii"), stored.encode("ascii"))
    except Exception:
        return False


def _reject_dupes(pairs: list) -> dict:
    d: dict = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate key")
        d[k] = v
    return d


def _reject_const(name: str) -> None:
    raise ValueError(f"non-finite constant {name}")


def receipt_bytes(receipt: dict) -> bytes:
    """Canonical on-disk form of a receipt (what ``verify_receipt_bytes`` expects)."""
    return canonical_bytes(receipt)


def verify_receipt_bytes(raw: bytes, key: bytes) -> bool:
    """Byte-exact verification: ANY single-byte change to a canonical receipt fails."""
    k = _require_key(key)
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_dupes,
                            parse_constant=_reject_const)
        if not isinstance(parsed, dict):
            return False
        if canonical_bytes(parsed) != raw:
            return False
        return verify_receipt(parsed, k)
    except Exception:
        return False


def sign_receipt(receipt: dict, key: bytes) -> dict:
    """Return a copy of ``receipt`` with its ``hmac`` field set."""
    out = {x: v for x, v in receipt.items() if x != "hmac"}
    out["hmac"] = sign(out, key)
    return out


# --------------------------------------------------------------------------- #
# Hash-chained log
# --------------------------------------------------------------------------- #
class LogCorruptError(RuntimeError):
    """Raised when appending would extend an already-corrupt log."""


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive lock: in-process RLock + cross-process fcntl.flock on <log>.lock."""
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        tl = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with tl:
        fd = None
        try:
            if fcntl is not None:
                fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def entry_hash_of(entry: dict) -> str:
    """sha256 of the canonical JSON of the entry without ``entry_hash``."""
    return sha256_hex(canonical_bytes({k: v for k, v in entry.items() if k != "entry_hash"}))


def _validate_log_input(entry: dict) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("log entry must be a dict")
    e = _native(dict(entry))
    for derived in ("seq", "prev_hash", "entry_hash"):  # assigned here, never trusted
        e.pop(derived, None)
    extra = set(e) - _LOG_INPUT_FIELDS
    if extra:
        raise ValueError(f"log entry fields not allowed: {sorted(extra)}")
    if not (isinstance(e.get("request_id"), str) and _REQID.fullmatch(e["request_id"])):
        raise ValueError("log entry needs a valid request_id")
    e.setdefault("timestamp_utc", utc_now())
    if not (isinstance(e["timestamp_utc"], str) and _TS.fullmatch(e["timestamp_utc"])):
        raise ValueError("log entry timestamp_utc malformed")
    sids = e.get("shard_ids")
    if not (isinstance(sids, list) and len(sids) <= _MAX_LIST
            and all(_is_int(s) and 0 <= s <= _MAX_SHARD_ID for s in sids)):
        raise ValueError("log entry shard_ids must be a short list of shard numbers")
    if not (isinstance(e.get("receipt_sha256"), str) and _HEX64.fullmatch(e["receipt_sha256"])):
        raise ValueError("log entry needs a 64-char hex receipt_sha256")
    return e


def _last_entry(p: Path) -> dict | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    raw = p.read_bytes()
    if not raw.endswith(b"\n"):
        raise LogCorruptError("log does not end with a newline (torn write?)")
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    if not lines:
        return None
    try:
        last = json.loads(lines[-1].decode("utf-8"))
        if not (_is_int(last["seq"]) and isinstance(last["entry_hash"], str)):
            raise ValueError
    except Exception as exc:
        raise LogCorruptError("last log entry is unreadable; refusing to extend the chain") from exc
    return last


def append_log_entry(entry: dict, path: str = DEFAULT_LOG_PATH) -> dict:
    """Append under a file lock and return the full stored entry (incl. seq)."""
    e = _validate_log_input(entry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(p):
        last = _last_entry(p)
        full = {
            "seq": 1 if last is None else last["seq"] + 1,
            "request_id": e["request_id"],
            "timestamp_utc": e["timestamp_utc"],
            "shard_ids": e["shard_ids"],
            "receipt_sha256": e["receipt_sha256"],
            "prev_hash": GENESIS_HASH if last is None else last["entry_hash"],
        }
        full["entry_hash"] = entry_hash_of(full)
        with open(p, "ab") as fh:
            fh.write(canonical_bytes(full) + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
    return full


def append_log(entry: dict, path: str = DEFAULT_LOG_PATH) -> str:
    """Spec 8.3 signature: append and return the new ``entry_hash``."""
    return append_log_entry(entry, path)["entry_hash"]


def verify_log(path: str = DEFAULT_LOG_PATH) -> tuple[bool, int | None]:
    """Recompute the chain. Returns (ok, first_bad_seq).

    ``first_bad_seq`` is the sequence number at which the chain first breaks:
    the edited entry itself, the position of a deleted entry, or - if an
    attacker recomputed an edited entry's own hash - the next entry whose
    ``prev_hash`` no longer matches. A missing/empty log is trivially valid.
    """
    p = Path(path)
    if not p.exists():
        return True, None
    lines = p.read_bytes().split(b"\n")
    while lines and not lines[-1].strip():
        lines.pop()
    prev = GENESIS_HASH
    for idx, raw in enumerate(lines):
        expected_seq = idx + 1
        try:
            entry = json.loads(raw.decode("utf-8"))
        except Exception:
            return False, expected_seq
        if not isinstance(entry, dict) or set(entry) != set(LOG_ENTRY_FIELDS):
            return False, expected_seq
        seq = entry["seq"]
        if not _is_int(seq) or seq != expected_seq:
            return False, expected_seq
        if entry["prev_hash"] != prev:
            return False, expected_seq
        if not isinstance(entry["entry_hash"], str) or entry_hash_of(entry) != entry["entry_hash"]:
            return False, expected_seq
        prev = entry["entry_hash"]
    return True, None


def read_log_entry(seq: int, path: str = DEFAULT_LOG_PATH) -> dict | None:
    """Return the stored entry with this seq, or None."""
    p = Path(path)
    if not p.exists():
        return None
    for raw in p.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            e = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(e, dict) and e.get("seq") == seq:
            return e
    return None


# --------------------------------------------------------------------------- #
# Receipt <-> log glue (circular-hash fix)
# --------------------------------------------------------------------------- #
def finalize_receipt(pre_receipt: dict, key: bytes, log_path: str = DEFAULT_LOG_PATH) -> dict:
    """Log the pre-log receipt, add log_seq/log_entry_hash, then HMAC-sign.

    Steps: validate pre-log receipt -> sha256 -> append log entry ->
    add ``log_seq`` + ``log_entry_hash`` -> sign.
    """
    k = _require_key(key)
    pre = build_receipt(**pre_log_receipt(pre_receipt))
    entry = append_log_entry(
        {
            "request_id": pre["request_id"],
            "timestamp_utc": pre["timestamp_utc"],
            "shard_ids": pre["shard_ids"],
            "receipt_sha256": receipt_sha256(pre),
        },
        log_path,
    )
    final = dict(pre)
    final["log_seq"] = entry["seq"]
    final["log_entry_hash"] = entry["entry_hash"]
    validate_receipt_fields(final)
    final["hmac"] = sign(final, k)
    return final


def verify_receipt_log_linkage(
    receipt: dict, path: str = DEFAULT_LOG_PATH, verify_chain: bool = True
) -> tuple[bool, str]:
    """Check that a final receipt really is the one recorded in the log.

    Re-derives the pre-log receipt hash and compares it with the log entry at
    ``log_seq``; also checks ``log_entry_hash``, request id, timestamp and
    shard ids, and (by default) the integrity of the whole chain.
    Returns (ok, reason).
    """
    try:
        seq = receipt.get("log_seq")
        eh = receipt.get("log_entry_hash")
        if not _is_int(seq) or not isinstance(eh, str):
            return False, "receipt has no log_seq / log_entry_hash"
        if verify_chain:
            ok, bad = verify_log(path)
            if not ok:
                return False, f"log chain is broken at seq {bad}"
        entry = read_log_entry(seq, path)
        if entry is None:
            return False, f"log has no entry with seq {seq} (log truncated or receipt forged)"
        if entry.get("entry_hash") != eh:
            return False, "log_entry_hash does not match the log entry"
        if entry.get("receipt_sha256") != receipt_sha256(receipt):
            return False, "receipt content does not match the hash recorded in the log"
        for field in ("request_id", "timestamp_utc", "shard_ids"):
            if entry.get(field) != receipt.get(field):
                return False, f"{field} differs between receipt and log entry"
        return True, "linked"
    except Exception as exc:  # never raise from a verifier
        return False, f"linkage check failed: {type(exc).__name__}"


def receipt_to_text(receipt: dict) -> str:
    """Printable receipt: counts, hashes, metrics, and how to verify. Not legal advice."""
    r = receipt or {}
    lines = [
        "ForgetMe erasure receipt",
        "This is not legal advice. The legal status of model weights under the DPDP Act is unsettled.",
        "The audit is empirical evidence from this attack, not a proof of forgetting.",
        "Synthetic demonstration. Not a medical device.",
        "",
        f"request_id: {r.get('request_id', '')}",
        f"timestamp_utc: {r.get('timestamp_utc', '')}",
        f"persons_erased: {r.get('persons_erased', '')}",
        f"rows_removed: {r.get('rows_removed', '')}",
        f"shard_ids: {r.get('shard_ids', '')}",
        f"full_retrain_seconds: {r.get('full_retrain_seconds', '')}",
        f"shard_retrain_seconds: {r.get('shard_retrain_seconds', '')}",
        f"speedup: {r.get('speedup', '')}",
    ]
    if r.get("full_retrain_serial_seconds") is not None:
        lines.append(f"full_retrain_serial_seconds: {r.get('full_retrain_serial_seconds')}")
    if r.get("speedup_vs_serial") is not None:
        lines.append(f"speedup_vs_serial: {r.get('speedup_vs_serial')}")
    lines.append(f"test_auroc_before: {r.get('test_auroc_before', '')}")
    lines.append(f"test_auroc_after: {r.get('test_auroc_after', '')}")
    lines.append(f"model_hash_before: {r.get('model_hash_before', '')}")
    lines.append(f"model_hash_after: {r.get('model_hash_after', '')}")
    lines.append(f"log_seq: {r.get('log_seq', '')}")
    lines.append(f"log_entry_hash: {r.get('log_entry_hash', '')}")
    lines.append(f"hmac: {r.get('hmac', '')}")
    guard = r.get("guard") if isinstance(r.get("guard"), dict) else {}
    if guard:
        lines.append(f"worst_subgroup: {guard.get('worst_subgroup')}")
        lines.append(f"max_subgroup_auroc_drop: {guard.get('max_subgroup_auroc_drop')}")
        flagged = guard.get("flagged_subgroups")
        if isinstance(flagged, dict) and flagged:
            lines.append(f"flagged_subgroups: {flagged}")
    lines += [
        "",
        "Verification: canonical JSON is sorted keys, separators (',', ':'), UTF-8,",
        "with the hmac field removed. The signature is HMAC-SHA256 with the key from",
        "the environment variable FORGETME_HMAC_KEY. verify_log checks the hash chain.",
        "A changed byte in the canonical receipt or in the log fails verification.",
    ]
    return "\n".join(lines) + "\n"


def receipt_to_html(receipt: dict) -> str:
    """Same content as receipt_to_text, escaped, for a browser download."""
    import html
    body = html.escape(receipt_to_text(receipt))
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>ForgetMe receipt</title></head><body><pre>"
        f"{body}</pre></body></html>\n"
    )
