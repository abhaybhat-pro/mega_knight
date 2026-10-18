"""Load every page and submit one erase. Numbers come from the modules, not placeholders.

  python scripts/smoke_ui.py            tiny random model (fast; overwrites data/ and models/ with toy files)
  python scripts/smoke_ui.py --real     the real generator (seed 42) and the real 8-shard model, as the demo uses
Run it with and without plotly installed to check the native-chart fallback.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.runtime.resources import configure_threads

configure_threads()

import numpy as np
import pandas as pd

from src.contracts import (
    CANARY_COL, COLUMNS, DATA_DIR, FEATURES, HELDOUT_CSV, ID_COL, LABEL_COL,
    MODELS_DIR, N_SHARDS, SEED, TEST_CSV, TRAIN_CSV,
)
from src.model.shards import train_all

if not os.environ.get("FORGETME_HMAC_KEY"):
    import secrets
    os.environ["FORGETME_HMAC_KEY"] = secrets.token_hex(16)


def _frame(n, id_start, rng):
    df = pd.DataFrame({name: rng.normal(size=n) for name in FEATURES})
    df["age"] = rng.integers(20, 80, n)
    df["sex"] = rng.integers(0, 2, n)
    df["village_code"] = rng.integers(0, 8, n)
    df["smoker"] = rng.integers(0, 2, n)
    for col in ("sym_fatigue", "sym_vision", "sym_thirst", "sym_chest_pain"):
        df[col] = rng.integers(0, 2, n)
    df["prior_visits"] = rng.integers(0, 5, n)
    df[LABEL_COL] = rng.integers(0, 2, n)
    df[ID_COL] = np.arange(id_start, id_start + n)
    df[CANARY_COL] = "none"
    return df[COLUMNS]


def _ensure_real_model() -> None:
    from src.data.generate import generate, read
    generate(out_dir=DATA_DIR)
    train_all(read(TRAIN_CSV, DATA_DIR), n_shards=N_SHARDS, n_workers=-1, models_dir=MODELS_DIR)


def _ensure_small_model() -> None:
    """A tiny on-disk ensemble so the erase click finishes inside the UI timeout."""
    data = Path(DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)
    marker = data / "smoke_marker.txt"
    if marker.is_file() and Path(MODELS_DIR, "manifest.json").is_file():
        return
    rng = np.random.default_rng(SEED)
    train = _frame(800, 1000, rng)
    test = _frame(400, 3_000_000, rng)
    held = _frame(200, 2_000_000, rng)
    train.to_csv(data / TRAIN_CSV, index=False)
    test.to_csv(data / TEST_CSV, index=False)
    held.to_csv(data / HELDOUT_CSV, index=False)
    train_all(train, n_shards=N_SHARDS, n_workers=1, models_dir=MODELS_DIR)
    marker.write_text("smoke\n", encoding="utf-8")


def main() -> int:
    real = "--real" in sys.argv
    (_ensure_real_model if real else _ensure_small_model)()
    import importlib.util
    print("plotly: installed" if importlib.util.find_spec("plotly") else "plotly: NOT installed (native chart fallback)")
    from streamlit.testing.v1 import AppTest

    files = [ROOT / "app" / "streamlit_app.py"] + sorted((ROOT / "app" / "pages").glob("*.py"))
    bad = 0
    for path in files:
        at = AppTest.from_file(str(path), default_timeout=120).run()
        print(("FAIL " if at.exception else "ok   ") + path.name)
        if at.exception:
            print(at.exception)
            bad += 1
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "3_Erase.py"), default_timeout=180).run()
    at.button[0].click().run()
    if at.exception:
        print(at.exception)
        bad += 1
    shown = [m.value for m in at.metric]
    good = (not at.exception) and "✔ VERIFIED" in shown
    print(("ok   " if good else "FAIL ") + "erase flow -> receipt + verified badge", shown)
    tb = [b for b in at.button if str(b.label).startswith("Tamper")]
    tamper_ok = False
    if tb:
        tb[0].click().run()
        tamper_ok = (not at.exception) and any("INVALID (expected" in str(i.value) for i in at.info)
    print(("ok   " if tamper_ok else "FAIL ") + "tamper test shows INVALID")
    bad += not tamper_ok
    return 1 if bad or not good else 0


if __name__ == "__main__":
    raise SystemExit(main())
