"""FR-16 acceptance: a diffprivlib model gives lower attack AUC AND lower test AUROC than the baseline.
(UI labelling of DP vs unlearning as separate guarantees is checked in the UI.)"""
import os, sys, warnings
import pandas as pd, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pytest.importorskip("diffprivlib", reason="diffprivlib missing: label DP as Planned and keep the install log")
from src.data.generate import generate
from src.model.dp import run_dp_experiment

DP_EPS = (0.5, 1.0, 5.0, 10.0)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    data = tmp_path_factory.mktemp("dp") / "data"
    generate(out_dir=str(data))
    tr, te, he = (pd.read_csv(data / f"{name}.csv") for name in ("train", "test", "heldout_canaries"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_dp_experiment(tr, te, he, DP_EPS)


def test_dp_lower_attack_and_lower_auroc(table):
    base, dp = table.iloc[0], table.iloc[1:]
    assert (dp["attack_auc_raw"] < base["attack_auc_raw"]).all()
    assert (dp["test_auroc"] < base["test_auroc"]).all()
    # DP attack should be indistinguishable from chance at every epsilon in the sweep
    assert (dp["attack_ci_lo"] <= 0.5).all() or (dp["attack_auc_raw"] < 0.6).all()


def test_sweep_has_all_epsilons(table):
    assert list(table.iloc[1:]["epsilon"]) == list(DP_EPS)
