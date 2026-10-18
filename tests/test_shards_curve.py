"""Tests for src/model/shards_curve.py (FR-14). Assertions are sanity checks only:
the 8.6 remaining-features section was not available, so no invented thresholds."""
import os
import sys

import pytest
import tempfile

pytestmark = pytest.mark.slow  # trains many forests; excluded from `make test`

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from src.contracts import TEST_CSV, TRAIN_CSV  # noqa: E402
from src.data.generate import generate, read  # noqa: E402
from src.model.shards_curve import cost_sentence, run_curve  # noqa: E402


def test_curve_files_and_values():
    d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    generate(out_dir=d)
    df = run_curve(read(TRAIN_CSV, d), read(TEST_CSV, d), out_dir=out)
    assert list(df["n_shards"]) == [1, 4, 8, 16]
    assert df["test_auroc"].between(0.5, 1.0, inclusive="neither").all()
    csv = pd.read_csv(os.path.join(out, "results.csv"))
    assert list(csv["n_shards"]) == [1, 4, 8, 16]
    png = os.path.join(out, "curve.png")
    assert os.path.getsize(png) > 1000
    s = cost_sentence(df)
    assert "8 shards" in s and "AUROC" in s and "cost" in s
    print(s)
