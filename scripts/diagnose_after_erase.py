#!/usr/bin/env python3
"""Diagnostics for tests/test_audit.py::test_after_erase (SC-13). Read-only: it never edits the
audit method, the thresholds or the test. It rebuilds the test fixture (generate, train 8 shards,
audit, simulated erase of EVERY member canary) in a temporary directory and prints:

  1. the per-shard post-erase audit table and the per-shard subset ensemble AUCs
  2. permutation nulls: the held-out canaries split into random halves, same scoring pipeline
  3. parity of member vs held-out canaries (features, label mean, attack-pattern mix)
  4. label-stratified AUCs and a label-matched (re-weighted) AUC
  5. raw versus shadow-calibrated post-erase AUC

    python scripts/diagnose_after_erase.py [--seed 42] [--json out.json] [--n-splits 200]

Evidence, not proof. Runs on whatever machine you are on; label the output with that machine.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.runtime.resources import configure_threads  # noqa: E402

configure_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.audit import mia  # noqa: E402
from src.contracts import FEATURES, N_SHARDS  # noqa: E402
from src.data.generate import generate, is_out_of_range  # noqa: E402
from src.model.shards import shard_of, train_all, train_shard  # noqa: E402


def erase_all_members(ss, train: pd.DataFrame, shards) -> object:
    new = dict(ss.models)
    for k in shards:
        part = train[train["person_id"].map(lambda p: shard_of(int(p), ss.n_shards)) == k]
        new[k] = train_shard(part[part["canary_type"] != "member"], k)
    return dataclasses.replace(ss, models=new)


def scores(ss, df, level, shard_id, sh):
    raw = mia.target_scores(ss, df, level, shard_id)
    S = sh["ens"] if level == "ensemble" else sh["shd"]
    idx = np.array([sh["index"][int(p)] for p in df["person_id"]])
    return mia.llr_scores(raw, idx, S, sh["inmask"]), raw


def weighted_auc(m: np.ndarray, n: np.ndarray, w: np.ndarray) -> float:
    gt = (m[:, None] > n[None, :]).astype(float) + 0.5 * (m[:, None] == n[None, :])
    return float((gt * w[None, :]).sum() / (len(m) * w.sum()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-workers", type=int, default=-1)
    a = ap.parse_args(argv)
    t0 = time.time()
    out: dict = {"seed": a.seed, "machine": os.uname().machine, "cores": os.cpu_count()}
    with tempfile.TemporaryDirectory() as tmp:
        data, models = os.path.join(tmp, "data"), os.path.join(tmp, "models")
        generate(seed=a.seed, out_dir=data)
        train = pd.read_csv(os.path.join(data, "train.csv"))
        held = pd.read_csv(os.path.join(data, "heldout_canaries.csv"))
        members = train[train["canary_type"] == "member"].reset_index(drop=True)
        ss = train_all(train, N_SHARDS, 1, models_dir=models)
        sh = mia.get_shadows(ss, data, n_jobs=a.n_workers)
        print(f"fixture ready in {time.time() - t0:.0f}s (seed {a.seed})", flush=True)
        ss_after = erase_all_members(ss, train, range(N_SHARDS))

        # 1. per-shard table after erase
        ens = mia.run_audit(ss_after, members, held, None, data_dir=data, n_workers=a.n_workers)["ensemble"]
        out["ensemble_after_all_members"] = {"auc": ens["attack_auc"], "ci": ens["auc_ci95"]}
        print(f"\n[1] ensemble after erasing all 400 members: AUC {ens['attack_auc']:.3f} "
              f"CI [{ens['auc_ci95'][0]:.3f}, {ens['auc_ci95'][1]:.3f}] raw {ens['attack_auc_raw']:.3f}")
        rows = []
        print("shard  n   shard AUC  CI                 raw    | subset ensemble AUC  CI")
        for k in range(N_SHARDS):
            r = mia.run_audit(ss_after, members, held, k, data_dir=data, n_workers=a.n_workers)
            s, e = r["shard"], r["ensemble"]
            rows.append(dict(shard=k, n=s["n_members"], shard_auc=s["attack_auc"], shard_ci=s["auc_ci95"],
                             shard_auc_raw=s["attack_auc_raw"], ens_auc=e["attack_auc"], ens_ci=e["auc_ci95"]))
            print(f"{k:>3} {s['n_members']:>4}   {s['attack_auc']:.3f}   [{s['auc_ci95'][0]:.3f}, {s['auc_ci95'][1]:.3f}]"
                  f"   {s['attack_auc_raw']:.3f}  |   {e['attack_auc']:.3f}   [{e['auc_ci95'][0]:.3f}, {e['auc_ci95'][1]:.3f}]")
        mean_shard = float(np.mean([r["shard_auc"] for r in rows]))
        miss_lo = sum(r["shard_ci"][1] < 0.5 for r in rows)
        miss_hi = sum(r["shard_ci"][0] > 0.5 for r in rows)
        out.update(per_shard=rows, mean_shard_auc=mean_shard, shard_ci_miss_low=miss_lo, shard_ci_miss_high=miss_hi)
        print(f"mean shard AUC {mean_shard:.3f}; shard CIs entirely below 0.5: {miss_lo}, above 0.5: {miss_hi}")

        # 2. permutation nulls (held-out split into random halves, same pipeline)
        rng = np.random.RandomState(a.seed + 99)
        n_h = len(held)
        ens_cal, _ = scores(ss_after, held, "ensemble", None, sh)
        shard_cal = [scores(ss_after, held, "shard", k, sh)[0] for k in range(N_SHARDS)]
        null_e, null_s = [], []
        for _ in range(a.n_splits):
            perm = rng.permutation(n_h)
            A, B = perm[: n_h // 2], perm[n_h // 2:]
            null_e.append(mia.auc(ens_cal[A], ens_cal[B]))
            for k in range(N_SHARDS):
                null_s.append(mia.auc(shard_cal[k][A], shard_cal[k][B]))
        out["null"] = dict(ensemble_mean=float(np.mean(null_e)), ensemble_sd=float(np.std(null_e)),
                           shard_mean=float(np.mean(null_s)), shard_sd=float(np.std(null_s)))
        print(f"\n[2] null (held-out halves, {a.n_splits} splits): ensemble {np.mean(null_e):.3f} +/- {np.std(null_e):.3f}; "
              f"shard level {np.mean(null_s):.3f} +/- {np.std(null_s):.3f}")

        # 3. parity member vs held-out
        print("\n[3] parity, members vs held-out canaries (400 each)")
        par = {}
        for c in FEATURES:
            ks = stats.ks_2samp(members[c], held[c])
            par[c] = dict(member_mean=float(members[c].mean()), heldout_mean=float(held[c].mean()), ks_p=float(ks.pvalue))
        ym, yh = int(members["needs_referral"].sum()), int(held["needs_referral"].sum())
        lab_p = float(stats.fisher_exact([[ym, 400 - ym], [yh, 400 - yh]])[1])
        par["label"] = dict(member_mean=ym / 400, heldout_mean=yh / 400, fisher_p=lab_p)
        pm = {"p0_young_obese": int(((members.bmi > 40) & (members.age < 25)).sum()),
              "p1_hba1c": int((members.hba1c > 11).sum())}
        ph = {"p0_young_obese": int(((held.bmi > 40) & (held.age < 25)).sum()),
              "p1_hba1c": int((held.hba1c > 11).sum())}
        out["parity"] = par
        out["pattern_counts"] = dict(member=pm, heldout=ph,
                                     out_of_range_member=int(is_out_of_range(members).sum()),
                                     out_of_range_heldout=int(is_out_of_range(held).sum()))
        for c in FEATURES:
            print(f"  {c:15s} member {par[c]['member_mean']:8.3f} held {par[c]['heldout_mean']:8.3f} KS p={par[c]['ks_p']:.3f}")
        print(f"  label mean      member {ym/400:8.3f} held {yh/400:8.3f} Fisher p={lab_p:.3f}")
        print(f"  pattern counts  member {pm} held {ph}")

        # 4. label-stratified and label-matched AUC (ensemble level, all 400 members)
        m_cal, m_raw = scores(ss_after, members, "ensemble", None, sh)
        h_cal, h_raw = scores(ss_after, held, "ensemble", None, sh)
        ym_v, yh_v = members["needs_referral"].values, held["needs_referral"].values
        strat = {}
        for y in (0, 1):
            strat[f"y={y}"] = dict(n_member=int((ym_v == y).sum()), n_held=int((yh_v == y).sum()),
                                   auc_cal=mia.auc(m_cal[ym_v == y], h_cal[yh_v == y]),
                                   auc_raw=mia.auc(m_raw[ym_v == y], h_raw[yh_v == y]))
        w = np.where(yh_v == 1, ym_v.mean() / yh_v.mean(), (1 - ym_v.mean()) / (1 - yh_v.mean()))
        matched = weighted_auc(m_cal, h_cal, w)
        rng2 = np.random.RandomState(a.seed + 5)
        boots = []
        for _ in range(300):
            bi, bj = rng2.randint(0, 400, 400), rng2.randint(0, 400, 400)
            boots.append(weighted_auc(m_cal[bi], h_cal[bj], w[bj]))
        out["label_stratified"] = strat
        out["label_matched_ensemble"] = dict(auc=matched, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                              unweighted_auc=mia.auc(m_cal, h_cal))
        print("\n[4] label-stratified ensemble AUC (members vs held-out, same label):")
        for k, v in strat.items():
            print(f"  {k}: n_member={v['n_member']} n_held={v['n_held']} cal AUC {v['auc_cal']:.3f} raw AUC {v['auc_raw']:.3f}")
        mm = out["label_matched_ensemble"]
        print(f"  label-matched (held-out re-weighted to the member label mix): AUC {mm['auc']:.3f} "
              f"CI [{mm['ci'][0]:.3f}, {mm['ci'][1]:.3f}]  (unweighted {mm['unweighted_auc']:.3f})")
        print(f"  mean score by label (raw logit-true): member y=0 {m_raw[ym_v==0].mean():.2f}, y=1 {m_raw[ym_v==1].mean():.2f}; "
              f"held y=0 {h_raw[yh_v==0].mean():.2f}, y=1 {h_raw[yh_v==1].mean():.2f}")

        # shard level, label-matched, per shard (members of that shard vs all held)
        sm = []
        for k in range(N_SHARDS):
            sel = np.array([shard_of(int(p), N_SHARDS) == k for p in members["person_id"]])
            mk = members[sel]
            c_m, _ = scores(ss_after, mk, "shard", k, sh)
            c_h = shard_cal[k]
            yk = mk["needs_referral"].values
            wk = np.where(yh_v == 1, yk.mean() / yh_v.mean(), (1 - yk.mean()) / (1 - yh_v.mean()))
            sm.append(dict(shard=k, auc=mia.auc(c_m, c_h), auc_label_matched=weighted_auc(c_m, c_h, wk)))
        out["shard_label_matched"] = sm
        print("  shard-level AUC, plain vs label-matched: " +
              ", ".join(f"{r['shard']}:{r['auc']:.3f}/{r['auc_label_matched']:.3f}" for r in sm))
        print(f"  mean plain {np.mean([r['auc'] for r in sm]):.3f}, mean label-matched "
              f"{np.mean([r['auc_label_matched'] for r in sm]):.3f}")

        # 6. matched-calibration null: held-out canaries only, split by the shard they hash to.
        # "Pseudo-members" are the held-out canaries that hash to shard k (matched calibration, like
        # real members); "pseudo-non-members" are the other held-out canaries. No real member is used,
        # so under a fair attack the AUC must be about 0.5 whatever the calibration does.
        pm_auc, pm_auc_rev = [], []
        hshard = np.array([shard_of(int(p), N_SHARDS) for p in held["person_id"]])
        for k in range(N_SHARDS):
            c = shard_cal[k]
            pm_auc.append(mia.auc(c[hshard == k], c[hshard != k]))
        out["held_only_matched_null"] = dict(per_shard=pm_auc, mean=float(np.mean(pm_auc)))
        print(f"\n[6] held-out only, hash-to-shard-k as pseudo-members (matched calibration): "
              f"mean shard AUC {np.mean(pm_auc):.3f} (per shard " + ", ".join(f"{v:.2f}" for v in pm_auc) + ")")

        # 7. matched on BOTH sides: members of shard k versus only those held-out canaries that
        # also hash to shard k (same shard's shadows calibrate both sets; about 50 vs 50, so noisy).
        both = []
        for k in range(N_SHARDS):
            sel = np.array([shard_of(int(p), N_SHARDS) == k for p in members["person_id"]])
            c_m, _ = scores(ss_after, members[sel], "shard", k, sh)
            both.append(mia.auc(c_m, shard_cal[k][hshard == k]))
        out["matched_both_sides"] = dict(per_shard=both, mean=float(np.mean(both)))
        print(f"\n[7] members(k) vs held-out(k), same-shard calibration on both sides: mean shard AUC "
              f"{np.mean(both):.3f} (per shard " + ", ".join(f"{v:.2f}" for v in both) + ")")

        # 5. calibration
        print(f"\n[5] raw vs calibrated post-erase, ensemble all 400: raw {ens['attack_auc_raw']:.3f}, calibrated {ens['attack_auc']:.3f}; "
              f"mean shard raw {np.mean([r['shard_auc_raw'] for r in rows]):.3f}, calibrated {mean_shard:.3f}")
        out["elapsed_s"] = time.time() - t0
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(f"\nelapsed {out['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
