"""Fixed contract for ForgetMe + Aegis-Z Guard (PRD section 8).

Everyone imports constants from here; nothing is hardcoded elsewhere.

Key names inside THRESHOLDS are not named by the PRD, so they are PROPOSED here
and become the contract the moment this file is pushed.

The function stubs at the bottom mirror 8.3 exactly and raise NotImplementedError.
Real implementations live in the module named in each stub's message. Import the
real function from its module, not from here. ``__all__`` exports constants and
dataclasses only, so ``from src.contracts import *`` never shadows a real function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # type hints only, keeps this module import-light
    from pandas import DataFrame
    from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------- 8.1 constants
SEED = 42
N_SHARDS = 8
N_PATIENTS = 20000  # train rows, INCLUDING member canaries
N_TEST = 2000
N_CANARY = 400      # per kind (member / heldout)

FEATURES = [
    "age", "sex", "bmi", "sbp", "dbp", "hba1c", "smoker",
    "sym_fatigue", "sym_vision", "sym_thirst", "sym_chest_pain",
    "village_code", "prior_visits",
]
ID_COL = "person_id"
LABEL_COL = "needs_referral"
CANARY_COL = "canary_type"
COLUMNS = [ID_COL] + FEATURES + [LABEL_COL, CANARY_COL]  # 8.2 column order
CANARY_TYPES = ("none", "member", "heldout")

N_VILLAGES = 8
RARE_VILLAGES = (6, 7)   # "two rare villages"
AGE_BANDS = ((18, 39), (40, 59), (60, 200))  # 18-39, 40-59, 60+

# person_id bases. Member/heldout bases are from the PRD; the other three are
# assumptions documented in docs/project/SPEC_CHANGES.md.
ID_BASE = {
    "train_ordinary": 100000,  # assumption: anything below 1,000,000
    "member": 1000000,
    "heldout": 2000000,
    "test": 3000000,
    "stream": 4000000,
}
N_STREAM_BASE = 10000

RF_PARAMS = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
                 bootstrap=True, n_jobs=1)

THRESHOLDS = {
    "referral_threshold": 0.5,
    "n_bootstrap": 1000,
    "ci_level": 0.95,
    "n_shadow_models": 16,
    "fpr_target": 0.01,          # TPR is reported at 1% FPR
    "n_bins": 10,                # quantile bins per numeric feature
    "psi_moderate": 0.10,
    "psi_major": 0.25,
    "min_subgroup_size": 200,
    "confidence_drop": 0.10,     # hold if mean confidence this far below baseline
    "subgroup_auroc_drop": 0.03,
    "batch_size": 500,
    "drift_start_batch": 10,
    "drift_shift": 1.5,
    "drift_village": 3,
    "drift_feature": "hba1c",
    "n_batches": 20,
    "probe_rows": 1000,
    "probe_decimals": 8,
}

HMAC_ENV_VAR = "FORGETME_HMAC_KEY"  # value never lives in code

# One path section. Resolved from this file, so a launch from another cwd
# still reads and writes the repo tree. Optional function arguments default
# to these absolute paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = str(_REPO_ROOT)
DATA_DIR = str(_REPO_ROOT / "data")
MODELS_DIR = str(_REPO_ROOT / "models")
SHADOW_DIR = str(_REPO_ROOT / "models" / "shadow")
LOG_PATH = str(_REPO_ROOT / "logs" / "erasure_log.jsonl")
BENCH_DIR = str(_REPO_ROOT / "bench")
DOCS_DIR = str(_REPO_ROOT / "docs")
TRAIN_CURRENT_PATH = str(_REPO_ROOT / "data" / "train_current.csv")
TRAIN_CSV, TEST_CSV = "train.csv", "test.csv"
HELDOUT_CSV, STREAM_CSV = "heldout_canaries.csv", "stream_base.csv"
MANIFEST_NAME = "manifest.json"
PERSON_MAP_NAME = "person_shard_map.json"


@dataclass
class ShardSet:
    models: "dict[int, RandomForestClassifier]"
    n_shards: int
    models_dir: str
    manifest: dict = field(default_factory=dict)


# Placeholders; the real classes live in src/guard/monitor.py.
Baseline = Any
BatchReport = Any

__all__ = [
    "SEED", "N_SHARDS", "N_PATIENTS", "N_TEST", "N_CANARY", "FEATURES", "ID_COL",
    "LABEL_COL", "CANARY_COL", "COLUMNS", "CANARY_TYPES", "N_VILLAGES",
    "RARE_VILLAGES", "AGE_BANDS", "ID_BASE", "N_STREAM_BASE", "RF_PARAMS",
    "THRESHOLDS", "HMAC_ENV_VAR", "DATA_DIR", "MODELS_DIR", "LOG_PATH",
    "TRAIN_CSV", "TEST_CSV", "HELDOUT_CSV", "STREAM_CSV", "MANIFEST_NAME",
    "PERSON_MAP_NAME", "REPO_ROOT", "SHADOW_DIR", "BENCH_DIR", "DOCS_DIR",
    "TRAIN_CURRENT_PATH", "ShardSet", "Baseline", "BatchReport",
]


# ------------------------------------------------- 8.3 signature stubs (NOT real)
def _todo(where: str):
    raise NotImplementedError(f"stub only: import the real function from {where}")


# src/data/generate.py  
def generate(seed: int = SEED) -> None:
    _todo("src.data.generate")


def drift_stream(base: "DataFrame", village: int = 3, feature: str = "hba1c",
                 shift: float = 1.5, start_batch: int = 10,
                 batch_size: int = 500, n_batches: int = 20) -> "Iterator[DataFrame]":
    _todo("src.data.generate")


# src/model/shards.py  
def shard_of(person_id: int, n_shards: int = N_SHARDS) -> int:
    _todo("src.model.shards")


def train_shard(df_shard: "DataFrame", shard_id: int,
                seed: int = SEED) -> "RandomForestClassifier":
    _todo("src.model.shards")


def train_all(train_df: "DataFrame", n_shards: int = N_SHARDS,
              n_workers: int = -1) -> ShardSet:
    _todo("src.model.shards")


def load(models_dir: str = MODELS_DIR) -> ShardSet:
    _todo("src.model.shards")


def predict(ss: ShardSet, X: "DataFrame") -> dict:
    _todo("src.model.shards")


def fingerprint(model, probe_X: "DataFrame") -> str:
    _todo("src.model.shards")


# src/erase/queue.py  
def erase(person_ids: "list[int]", request_id: str, ss: ShardSet,
          train_df: "DataFrame") -> dict:
    _todo("src.erase.queue")


def append_log(entry: dict, path: str = LOG_PATH) -> str:
    _todo("src.erase.queue")


def verify_log(path: str = LOG_PATH) -> "tuple[bool, int | None]":
    _todo("src.erase.queue")


# src/audit/mia.py  
def run_audit(ss: ShardSet, members: "DataFrame", nonmembers: "DataFrame",
              shard_id: "int | None" = None) -> dict:
    _todo("src.audit.mia")


# src/guard/monitor.py  
def fit_baseline(train_df: "DataFrame") -> Baseline:
    _todo("src.guard.monitor")


def score_batch(ss: ShardSet, batch: "DataFrame", base: Baseline) -> BatchReport:
    _todo("src.guard.monitor")


def subgroup_auroc(ss: ShardSet, test_df: "DataFrame") -> dict:
    _todo("src.guard.monitor")


# src/report/receipt.py  
def build_receipt(**fields) -> dict:
    _todo("src.report.receipt")


def sign(receipt: dict, key: bytes) -> str:
    _todo("src.report.receipt")


def verify_receipt(receipt: dict, key: bytes) -> bool:
    _todo("src.report.receipt")


# src/report/narrate.py  
def template_report(receipt: dict) -> str:
    _todo("src.report.narrate")


def granite_report(receipt: dict) -> str:
    _todo("src.report.narrate")


def validate_numbers(text: str, receipt: dict) -> "tuple[bool, list[str]]":
    _todo("src.report.narrate")
