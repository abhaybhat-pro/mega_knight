"""End-to-end demo driver (FR-11). Called by scripts/run_demo.sh.

Steps (each prints one line):
  1 generate   2 train   3 baseline audit   4 guard drift stream (+ no-drift stream)
  5 erase one member-canary batch (all member canaries that live in one shard)
  6 after-audit (read from the receipt)   7 post-erase subgroup check
  8 report (Granite if configured, else template; numbers validated)   9 verification

It only calls functions from PRD 8.3. Where a return shape is not fixed by the PRD
(subgroup_auroc) it reads fields defensively and says so instead of crashing.
Needs env FORGETME_HMAC_KEY (run_demo.sh sets a throwaway one). Prints the receipt path last.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.runtime.resources import configure_threads  # noqa: E402

configure_threads()

import pandas as pd  # noqa: E402

from src.contracts import (CANARY_COL, DATA_DIR, HELDOUT_CSV, HMAC_ENV_VAR, ID_COL,  # noqa: E402
                           LOG_PATH, MODELS_DIR, N_SHARDS, SEED, STREAM_CSV, TEST_CSV,
                           THRESHOLDS, TRAIN_CSV)

DEMO_SHARD = int(os.environ.get("DEMO_SHARD", "3")) % N_SHARDS
RECEIPT_DIR = os.path.join(ROOT, "receipts")


def step(n: int, name: str) -> float:
    print(f"[{n}/9] {name} ...", flush=True)
    return time.perf_counter()


def done(t0: float, msg: str = "") -> None:
    print(f"      done in {time.perf_counter() - t0:.1f}s {msg}".rstrip(), flush=True)


def need(modname: str):
    try:
        return __import__(modname, fromlist=["_"])
    except ImportError as e:
        print(f"ERROR: {modname} is not available yet ({e}). It is owned by another dev; "
              "see docs/ibm_status.md for the build status.", file=sys.stderr)
        raise SystemExit(3)


def _attr(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def summarise_guard(monitor, reports: list, quiet_reports: list) -> dict:
    """Guard stream results as COUNTS ONLY, ready for the receipt (guard.drift_flags,
    guard.held_decisions). Person ids stay inside monitor.stream_counts and are never printed."""
    counts = monitor.stream_counts(reports)
    quiet = monitor.stream_counts(quiet_reports)
    adv_quiet = monitor.stream_advisories(quiet_reports)
    print(f"      drift stream: {counts['drift_flags']} village(s) reached the major drift line, "
          f"{counts['held_decisions']} decision(s) held for human review")
    print(f"      no-drift stream: {quiet['held_decisions']} holds and {quiet['drift_flags']} village flags; "
          f"advisories (never added to the hold list): {adv_quiet or 'none'}")
    return counts


def auroc_drop(before, after) -> float | None:
    """Largest per-subgroup AUROC drop, nested or flat."""
    try:
        from src.guard.monitor import flatten_subgroup_auroc
        b, a = flatten_subgroup_auroc(before), flatten_subgroup_auroc(after)
        common = [k for k in b if k in a]
        return max(b[k] - a[k] for k in common) if common else None
    except (AttributeError, TypeError, ValueError, ImportError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-guard", action="store_true", help="skip the guard stream")
    a = ap.parse_args(argv)

    if not os.environ.get(HMAC_ENV_VAR):
        print(f"ERROR: {HMAC_ENV_VAR} is not set (scripts/run_demo.sh sets a throwaway one).",
              file=sys.stderr)
        return 2
    key = os.environ[HMAC_ENV_VAR].encode()
    os.makedirs(RECEIPT_DIR, exist_ok=True)

    gen = need("src.data.generate")
    shards = need("src.model.shards")
    mia = need("src.audit.mia")
    queue = need("src.erase.queue")
    receipt_mod = need("src.report.receipt")
    narrate = need("src.report.narrate")
    monitor = None if a.no_guard else need("src.guard.monitor")

    # rotate any previous log so the demo starts a fresh chain (kept, never deleted)
    if os.path.exists(LOG_PATH):
        bak = f"{LOG_PATH}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
        shutil.move(LOG_PATH, bak)
        print(f"      previous log moved to {os.path.relpath(bak, ROOT)}")
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

    # 1 generate
    t = step(1, f"generate synthetic data (seed {SEED})")
    gen.generate(SEED)
    train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV))
    test = pd.read_csv(os.path.join(DATA_DIR, TEST_CSV))
    held = pd.read_csv(os.path.join(DATA_DIR, HELDOUT_CSV))
    stream_base = pd.read_csv(os.path.join(DATA_DIR, STREAM_CSV))
    done(t, f"({len(train)} train rows, {len(test)} test rows)")

    # 2 train
    t = step(2, f"train {N_SHARDS} shards")
    ss = shards.train_all(train, N_SHARDS, -1)
    done(t)

    # 3 baseline audit on the shard that will be erased from
    members = train[train[CANARY_COL] == "member"]
    sid = shards.assign_shards(train[ID_COL], N_SHARDS)
    t = step(3, f"baseline audit (shard {DEMO_SHARD} and whole model)")
    base_audit = mia.run_audit(ss, members, held, shard_id=DEMO_SHARD)
    done(t)
    for scope in ("ensemble", "shard"):
        blk = _attr(base_audit, scope, default={}) or {}
        print(f"      baseline {scope}: attack_auc={blk.get('attack_auc')} "
              f"tpr@1%fpr={blk.get('tpr_at_1pct_fpr')}")

    # 4 guard
    guard = None
    sub_before = None
    if monitor is None:
        step(4, "guard stream (skipped by --no-guard)")
    else:
        t = step(4, "guard: drift stream and no-drift stream")
        base = monitor.fit_baseline(train, ss=ss, test_df=test)
        reports = [monitor.score_batch(ss, b, base) for b in gen.drift_stream(stream_base)]
        quiet_base = monitor.fit_baseline(train, ss=ss, test_df=test)
        quiet = [monitor.score_batch(ss, b, quiet_base)
                 for b in gen.drift_stream(stream_base, shift=0.0)]
        guard = summarise_guard(monitor, reports, quiet)
        sub_before = monitor.subgroup_auroc(ss, test)
        done(t)

    # 5 erase: every member canary that lives in DEMO_SHARD, as one request
    ids = [int(p) for p in members.loc[sid[members.index] == DEMO_SHARD, ID_COL]]
    t = step(5, f"erase {len(ids)} member canaries in shard {DEMO_SHARD} (one request)")
    if not ids:
        print("ERROR: no member canaries in the demo shard", file=sys.stderr)
        return 1
    # guard is None with --no-guard: the receipt then says "not recorded" for those two counts
    receipt = queue.erase(ids, "REQ-0001", ss, train, guard_summary=guard)
    done(t)
    g = receipt.get("guard") or {}
    print(f"      receipt guard counts: drift_flags={g.get('drift_flags')} "
          f"held_decisions={g.get('held_decisions')} (counts only, no person ids)")
    rpath = os.path.join(RECEIPT_DIR, f"{receipt.get('request_id', 'REQ-0001')}.json")
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)  # counts and metrics only

    # 6 after-audit, from the receipt
    step(6, "after-audit (from the receipt)")
    for scope in ("ensemble", "shard"):
        blk = (receipt.get("audit_after") or {}).get(scope) or {}
        print(f"      after {scope}: attack_auc={blk.get('attack_auc')} "
              f"tpr@1%fpr={blk.get('tpr_at_1pct_fpr')}")
    attack = {"before": base_audit, "after": receipt.get("audit_after")}
    with open(os.path.join(RECEIPT_DIR, "demo_attack_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(attack, fh, indent=1, sort_keys=True, default=str)  # diff across runs

    # 7 post-erase subgroup check (printed here; the receipt's own guard block is built by the erase module)
    if monitor is not None and sub_before is not None:
        t = step(7, "post-erase subgroup AUROC check")
        ss_after = shards.load(MODELS_DIR)
        drop = auroc_drop(sub_before, monitor.subgroup_auroc(ss_after, test))
        limit = THRESHOLDS["subgroup_auroc_drop"]
        if drop is None:
            print("      subgroup_auroc shape not recognised; drop not computed here")
        else:
            print(f"      max subgroup AUROC drop {drop:.4f} (limit {limit}): "
                  f"{'OK' if drop <= limit else 'EXCEEDS LIMIT'}")
        done(t)
    else:
        step(7, "post-erase subgroup check (skipped)")

    # 8 report
    t = step(8, "plain-language report")
    text, info = narrate.generate_report_with_info(receipt)
    ok_nums, bad = narrate.validate_numbers(text, receipt)
    rep_path = os.path.join(RECEIPT_DIR, f"{receipt.get('request_id', 'REQ-0001')}_report.txt")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"      writer: {info['writer']}" + (f" ({info['fallback_reason']})"
                                                if info["fallback_reason"] else ""))
    print(f"      numbers validator: {'PASS' if ok_nums else 'FAIL ' + str(bad)}")
    done(t)

    # 9 verification
    step(9, "verification (receipt signature, log chain)")
    sig_ok = receipt_mod.verify_receipt(receipt, key)
    log_ok, first_bad = receipt_mod.verify_log(LOG_PATH)
    print(f"      receipt signature: {'OK' if sig_ok else 'FAIL'}")
    print(f"      erasure log chain: {'OK' if log_ok else f'FAIL at seq {first_bad}'}")

    print(f"REPORT:  {os.path.relpath(rep_path, ROOT)}")
    print(f"RECEIPT: {os.path.relpath(rpath, ROOT)}")
    return 0 if (sig_ok and log_ok and ok_nums) else 1


if __name__ == "__main__":
    raise SystemExit(main())
