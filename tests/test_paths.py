"""Paths come from contracts.py's location, not the process cwd."""
from pathlib import Path

from src.contracts import DATA_DIR, LOG_PATH, MODELS_DIR, REPO_ROOT, TRAIN_CSV
from src.data.generate import generate, read


def test_contracts_paths_are_absolute_and_anchored():
    root = Path(REPO_ROOT)
    assert Path(DATA_DIR).is_absolute()
    assert Path(DATA_DIR) == root / "data"
    assert Path(MODELS_DIR) == root / "models"
    assert Path(LOG_PATH) == root / "logs" / "erasure_log.jsonl"


def test_generate_out_dir_is_independent_of_cwd(tmp_path, monkeypatch):
    out = tmp_path / "clinic"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    generate(out_dir=str(out))
    assert not (elsewhere / "data").exists()
    df = read(TRAIN_CSV, str(out))
    assert len(df) > 0
    assert (out / TRAIN_CSV).is_file()
