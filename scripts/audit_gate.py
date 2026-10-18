#!/usr/bin/env python3
"""Hour-3 go/no-go gate. Trains fresh shards, audits BEFORE erase, prints PASS/FAIL.

Gate (frozen, never edit): CALIBRATED shard AUC >= 0.90, CALIBRATED ensemble AUC >= 0.62, lower 95% CI bound > 0.5 on both.
The floors are checked against the calibrated attack AUC (the audit's headline number), never the raw AUC.
The raw AUC is printed beside it as a sanity column: it shows whether the signal exists without calibration.
Shard-level is judged on the WORST of the 8 shards (stricter than a mean); the mean is also printed.
These two floors live here and are imported by tests/test_audit.py so they cannot drift.
They are not THRESHOLDS keys (SC-3).
Run from the repo root:  python scripts/audit_gate.py [--n-workers -1]
NOTE: train_all writes models/ (deterministic, so overwriting is harmless).
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.runtime.resources import configure_threads
configure_threads()
import pandas as pd
from src.contracts import DATA_DIR, N_SHARDS
from src.model.shards import train_all
from src.audit.mia import run_audit
from src.audit.secondary import run_secondary_audit

SHARD_MIN = 0.90
ENS_MIN = 0.62


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--n-workers", type=int, default=-1)
    a = ap.parse_args(argv)
    print(f"host={os.uname().machine}; cores={os.cpu_count()}; any number from a host that is not s390x is a "
          f"rehearsal number, never an s390x result", flush=True)
    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    held = pd.read_csv(os.path.join(a.data_dir, "heldout_canaries.csv"))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    members = train[train["canary_type"] == "member"]
    t0 = time.time()
    ss = train_all(train, N_SHARDS, a.n_workers)
    print(f"trained {ss.n_shards} shards in {time.time()-t0:.1f}s")
    t0 = time.time()
    ens = run_audit(ss, members, held, None, data_dir=a.data_dir)["ensemble"]
    print(f"audit (incl. shadow build/cache load) {time.time()-t0:.1f}s\n")
    rows = []
    for k in range(ss.n_shards):
        r = run_audit(ss, members, held, k, data_dir=a.data_dir)["shard"]
        rows.append(r)
        print(f"shard {k}: n={r['n_members']:3d} AUC cal={r['attack_auc']:.3f} CI={r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f} "
              f"raw={r['attack_auc_raw']:.3f} TPR@1%FPR={r['tpr_at_1pct_fpr']:.3f}")
    print(f"\nensemble: AUC cal={ens['attack_auc']:.3f} CI={ens['auc_ci95'][0]:.3f}-{ens['auc_ci95'][1]:.3f} "
          f"raw={ens['attack_auc_raw']:.3f} (CI {ens['auc_ci95_raw'][0]:.3f}-{ens['auc_ci95_raw'][1]:.3f}) "
          f"TPR@1%FPR={ens['tpr_at_1pct_fpr']:.3f} (CI {ens['tpr_ci95'][0]:.3f}-{ens['tpr_ci95'][1]:.3f})")
    sec = run_secondary_audit(ss, train, test)
    print(f"secondary (500 train vs 500 test, raw): ensemble AUC={sec['ensemble']['attack_auc_raw']:.3f} "
          f"CI={sec['ensemble']['auc_ci95_raw'][0]:.3f}-{sec['ensemble']['auc_ci95_raw'][1]:.3f}")
    worst = min(rows, key=lambda r: r["attack_auc"])
    mean_auc = sum(r["attack_auc"] for r in rows) / len(rows)
    checks = {
        f"shard worst calibrated AUC >= {SHARD_MIN} (worst={worst['attack_auc']:.3f}, mean={mean_auc:.3f})": worst["attack_auc"] >= SHARD_MIN,
        f"shard lower CI > 0.5 on every shard (min={min(r['auc_ci95'][0] for r in rows):.3f})": all(r["auc_ci95"][0] > 0.5 for r in rows),
        f"ensemble calibrated AUC >= {ENS_MIN} ({ens['attack_auc']:.3f})": ens["attack_auc"] >= ENS_MIN,
        f"ensemble lower CI > 0.5 ({ens['auc_ci95'][0]:.3f})": ens["auc_ci95"][0] > 0.5,
    }
    print("\nfloor check uses the CALIBRATED AUC; raw shown side by side")
    print(f"{'scope':10s} {'calibrated':>11s} {'raw':>7s} {'floor':>6s}")
    for i, r in enumerate(rows):
        print(f"{'shard '+str(i):10s} {r['attack_auc']:11.3f} {r['attack_auc_raw']:7.3f} {SHARD_MIN:6.2f}")
    print(f"{'ensemble':10s} {ens['attack_auc']:11.3f} {ens['attack_auc_raw']:7.3f} {ENS_MIN:6.2f}")
    print()
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    ok = all(checks.values())
    print("\nGATE:", "PASS" if ok else "FAIL")
    if not ok:
        print("Escalation ladder (8.5), IN ORDER, thresholds are frozen: 1) bug hunt 2) more extreme canaries / 800 members "
              "3) max_features=None (team decision) 4) AUC-vs-shard-count experiment. Report to the team.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
