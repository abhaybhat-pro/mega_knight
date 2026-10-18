"""Accuracy cost of sharding (FR-14).

Test AUROC of the mean-of-shards ensemble for N = 1, 4, 8, 16 shards.
Writes bench/shards_curve/results.csv (8.6 FR-14 name; kept in its own folder so it
cannot collide with FR-10's bench/results.csv) and a labelled PNG (distinct markers + value labels,
so it does not rely on colour alone). Nothing is saved to models/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.contracts import DATA_DIR, LABEL_COL, SEED, TEST_CSV, TRAIN_CSV, ShardSet
from src.model.shards import fit_models, predict

SHARD_COUNTS = (1, 4, 8, 16)
BENCH_DIR = os.path.join("bench", "shards_curve")


def run_curve(train_df: pd.DataFrame, test_df: pd.DataFrame,
              shard_counts=SHARD_COUNTS, n_workers: int = -1,
              out_dir: str = BENCH_DIR, seed: int = SEED) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for n in shard_counts:
        models = fit_models(train_df, n_shards=n, n_workers=n_workers, seed=seed)
        ss = ShardSet(models=models, n_shards=n, models_dir="")
        auc = roc_auc_score(test_df[LABEL_COL], predict(ss, test_df)["proba"])
        rows.append({"n_shards": n, "rows_per_shard": len(train_df) // n,
                     "test_auroc": round(float(auc), 6)})
    df = pd.DataFrame(rows)
    base = df.loc[df["n_shards"] == df["n_shards"].min(), "test_auroc"].iloc[0]
    df["auroc_delta_vs_fewest"] = (df["test_auroc"] - base).round(6)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False, lineterminator="\n")
    _plot(df, os.path.join(out_dir, "curve.png"))
    return df


def _plot(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    x = np.arange(len(df))
    ax.plot(x, df["test_auroc"], marker="o", markersize=8, linewidth=2,
            color="black", label="Ensemble test AUROC (mean of shards)")
    for xi, (n, a) in enumerate(zip(df["n_shards"], df["test_auroc"])):
        ax.annotate(f"{a:.3f}", (xi, a), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in df["n_shards"]])
    ax.set_xlabel("Number of shards (each trained on 1/N of the rows)")
    ax.set_ylabel("AUROC on test.csv (2,000 synthetic rows)")
    ax.set_title("Accuracy cost of sharding (synthetic data)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cost_sentence(df: pd.DataFrame, chosen: int = 8) -> str:
    one = df.loc[df["n_shards"] == df["n_shards"].min()].iloc[0]
    sel = df.loc[df["n_shards"] == chosen].iloc[0]
    delta = sel["test_auroc"] - one["test_auroc"]
    verdict = ("no accuracy cost observed" if delta >= 0
               else f"accuracy cost of {-delta:.3f} AUROC")
    return (f"Sharding result: {verdict}. Going from {int(one['n_shards'])} shard "
            f"(AUROC {one['test_auroc']:.3f}) to {chosen} shards "
            f"(AUROC {sel['test_auroc']:.3f}) changes test AUROC by {delta:+.3f} "
            f"on synthetic data.")


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV))
    te = pd.read_csv(os.path.join(DATA_DIR, TEST_CSV))
    result = run_curve(tr, te)
    print(result.to_string(index=False))
    print(cost_sentence(result))
