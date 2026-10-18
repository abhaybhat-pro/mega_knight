"""FR-5 acceptance (spec 8.6). Thresholds are FROZEN; if a test misses, follow the 8.5 ladder, don't edit them.

Fixtures use the real generate() and train_all().
After-erase is SIMULATED here (retrain shard on its data minus the erased canaries) so this file
tests the audit code independently of the erase flow.
"""
import dataclasses, os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.contracts import N_SHARDS
from src.data.generate import generate
from src.model.shards import shard_of, train_all, train_shard
from src.audit.mia import run_audit
from src.audit.secondary import run_secondary_audit
from scripts.audit_gate import ENS_MIN, SHARD_MIN

pytestmark = pytest.mark.slow  # trains many forests; excluded from `make test`

PREREG_SHARD = 3     # pre-registered shard for the after-erase assertion (not chosen post hoc)


def _audit(env, ss, members, nonmembers, shard_id):
    return run_audit(ss, members, nonmembers, shard_id, data_dir=env["data_dir"])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit")
    data = root / "data"
    models = root / "models"
    generate(out_dir=str(data))
    train = pd.read_csv(data / "train.csv")
    held = pd.read_csv(data / "heldout_canaries.csv")
    test = pd.read_csv(data / "test.csv")
    ss = train_all(train, N_SHARDS, 1, models_dir=str(models))
    members = train[train["canary_type"] == "member"]
    d = str(data)
    before = {"ens": run_audit(ss, members, held, None, data_dir=d)["ensemble"],
              "shard": {k: run_audit(ss, members, held, k, data_dir=d)["shard"] for k in range(N_SHARDS)}}
    return dict(train=train, held=held, test=test, ss=ss, members=members, before=before, data_dir=d)


def _erase(env, shards):
    """Simulate erase: retrain each shard in `shards` without its member canaries."""
    tr, ss = env["train"], env["ss"]
    new = dict(ss.models)
    for k in shards:
        d = tr[tr["person_id"].map(lambda p: shard_of(int(p), ss.n_shards)) == k]
        new[k] = train_shard(d[d["canary_type"] != "member"], k)
    return dataclasses.replace(ss, models=new)


def test_bug_hunt_preconditions(env):          # ladder step 1, kept as a permanent regression test
    tr, held = env["train"], env["held"]
    assert (tr["canary_type"] == "member").sum() == 400 and len(held) == 400
    assert not set(held["person_id"]) & set(tr["person_id"])
    assert tr["person_id"].is_unique
    assert set(tr["needs_referral"]) <= {0, 1}


def test_before_erase_gate(env):
    b = env["before"]
    for k, r in b["shard"].items():
        assert r["attack_auc"] >= SHARD_MIN, f"shard {k} AUC {r['attack_auc']:.3f}"
        assert r["auc_ci95"][0] > 0.5
    e = b["ens"]
    assert e["attack_auc"] >= ENS_MIN, e["attack_auc"]
    assert e["auc_ci95"][0] > 0.5
    # raw ensemble must also carry signal (guards against calibration producing the effect alone)
    assert e["auc_ci95_raw"][0] > 0.5


def test_shadow_cache_reused_and_audit_deterministic(env):
    ss = env["ss"]
    files = sorted(os.listdir(os.path.join(ss.models_dir, "shadow")))
    assert len(files) == 1
    p = os.path.join(ss.models_dir, "shadow", files[0]); m0 = os.path.getmtime(p)
    a = _audit(env, ss, env["members"], env["held"], PREREG_SHARD)
    b = _audit(env, ss, env["members"], env["held"], PREREG_SHARD)
    assert a == b                                   # deterministic given SEED
    assert os.path.getmtime(p) == m0                # cache untouched
    assert len(os.listdir(os.path.join(ss.models_dir, "shadow"))) == 1


def test_after_erase(env):
    ss_after = _erase(env, range(N_SHARDS))         # all member canaries erased -> n=400 for the ensemble band
    held, members = env["held"], env["members"]
    ens = _audit(env, ss_after, members, held, None)["ensemble"]
    lo, hi = ens["auc_ci95"]
    assert lo <= 0.5 <= hi, ens["auc_ci95"]
    assert abs(ens["attack_auc"] - 0.5) <= 0.05, ens["attack_auc"]
    # SC-13 (lead-approved): the two SUBSET-level checks are one-sided. The privacy claim is "the attack is
    # not significantly better than chance", i.e. CI lower bound <= 0.5. Shard-level calibration is
    # asymmetric (held-out canaries use the statistics of the shard they hash to, members their own),
    # which puts the post-erase shard AUC at or slightly below 0.5 (9 seeds: ensemble mean 0.494,
    # calibrated shard mean 0.478, raw mean 0.499; matched held-out-only pseudo-members reproduce it).
    # A two-sided "CI contains 0.5" therefore fails on the low side without any leak.
    r = _audit(env, ss_after, members, held, PREREG_SHARD)
    assert r["shard"]["auc_ci95"][0] <= 0.5, r["shard"]["auc_ci95"]
    assert r["ensemble"]["auc_ci95"][0] <= 0.5, r["ensemble"]["auc_ci95"]
    # Power check: the SAME one-sided test must FAIL on the before-erase audit of the same shard and
    # ensemble subset (lower bound > 0.5), so it would detect a leak.
    pre = _audit(env, env["ss"], members, held, PREREG_SHARD)
    assert not (pre["shard"]["auc_ci95"][0] <= 0.5), pre["shard"]["auc_ci95"]
    assert not (pre["ensemble"]["auc_ci95"][0] <= 0.5), pre["ensemble"]["auc_ci95"]
    # information only (NOT asserted; ~1 in 3 of 8 independent 95% CIs would miss 0.5 by chance alone)
    cover = sum(_audit(env, ss_after, members, held, k)["shard"]["auc_ci95"][0] <= 0.5 <=
                _audit(env, ss_after, members, held, k)["shard"]["auc_ci95"][1] for k in range(N_SHARDS))
    print(f"after-erase shard CIs containing 0.5: {cover}/{N_SHARDS}")


def test_single_shard_erase_ensemble_is_informational(env):
    """Spec-faithful single-shard erase. The 0.05 band is NOT asserted: with ~50 members vs 400 the
    null AUC has sd ~0.04, so |AUC-0.5|<=0.05 fails ~20% of the time even for a perfect erase."""
    ss_after = _erase(env, [PREREG_SHARD])
    r = _audit(env, ss_after, env["members"], env["held"], PREREG_SHARD)
    b = env["before"]["shard"][PREREG_SHARD]
    assert r["shard"]["attack_auc"] < b["attack_auc"] - 0.3
    print(f"single-shard erase: shard AUC {b['attack_auc']:.3f} -> {r['shard']['attack_auc']:.3f}; "
          f"ensemble AUC {r['ensemble']['attack_auc']:.3f} CI {r['ensemble']['auc_ci95']}")


def test_schema_shape(env):
    r = _audit(env, env["ss"], env["members"], env["held"], PREREG_SHARD)
    keys = {"attack_auc", "auc_ci95", "tpr_at_1pct_fpr", "canary_percentile_median", "tpr_ci95", "attack_auc_raw"}
    assert keys <= set(r["ensemble"]) and keys <= set(r["shard"]) and r["shard_detail"]["shard_id"] == PREREG_SHARD
    r0 = _audit(env, env["ss"], env["members"], env["held"], None)
    assert r0["shard"] is None and r0["shard_detail"] is None
    assert 0 <= r["shard"]["canary_percentile_median"] <= 100


def test_secondary_weaker_than_canaries(env):
    s = run_secondary_audit(env["ss"], env["train"], env["test"])
    assert s["ensemble"]["attack_auc_raw"] < env["before"]["ens"]["attack_auc_raw"]
    assert s["shard"]["attack_auc_raw"] < env["before"]["shard"][PREREG_SHARD]["attack_auc_raw"]
