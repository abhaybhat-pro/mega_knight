#!/usr/bin/env python3
"""PRD 8.3 / 8.1 contract check, plus repository structure checks. Run before every merge.

Exit 0: no errors for the requested wave.
Exit 1: at least one error.

Warnings are printed and do not fail the process. A warning is still a bug
report: it means a module hardcoded a value that currently matches the contract.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.resources import configure_threads

configure_threads()

# Spec function -> module that must provide the real implementation.
# These names and their leading parameters are the stubs in src/contracts.py.
IMPLEMENTATIONS = {
    "generate": "src.data.generate",
    "drift_stream": "src.data.generate",
    "shard_of": "src.model.shards",
    "train_shard": "src.model.shards",
    "train_all": "src.model.shards",
    "load": "src.model.shards",
    "predict": "src.model.shards",
    "fingerprint": "src.model.shards",
    "erase": "src.erase.queue",
    "append_log": "src.erase.queue",
    "verify_log": "src.erase.queue",
    "run_audit": "src.audit.mia",
    "fit_baseline": "src.guard.monitor",
    "score_batch": "src.guard.monitor",
    "subgroup_auroc": "src.guard.monitor",
    "build_receipt": "src.report.receipt",
    "sign": "src.report.receipt",
    "verify_receipt": "src.report.receipt",
    "template_report": "src.report.narrate",
    "granite_report": "src.report.narrate",
    "validate_numbers": "src.report.narrate",
}

WAVE_FUNCTIONS = {
    0: [],
    1: [
        "generate", "drift_stream", "shard_of", "train_shard", "train_all", "load",
        "predict", "fingerprint", "run_audit", "build_receipt", "sign", "verify_receipt",
        "append_log", "verify_log",
    ],
    2: None,  # filled below: wave 1 plus erase, guard, report
    3: None,
}
WAVE_FUNCTIONS[2] = WAVE_FUNCTIONS[1] + [
    "erase", "fit_baseline", "score_batch", "subgroup_auroc",
    "template_report", "granite_report", "validate_numbers",
]
WAVE_FUNCTIONS[3] = list(IMPLEMENTATIONS)

ALLOWED_THIRD_PARTY = {
    "numpy", "pandas", "sklearn", "scipy", "joblib", "streamlit", "matplotlib", "pytest",
}
# import-name -> path fragments where that import is allowed
EXTRA_IMPORTS = {
    "diffprivlib": ("src/model/dp.py", "tests/test_dp.py"),
    "skl2onnx": ("src/serving/", "tests/test_onnx.py"),
    "onnx": ("src/serving/", "tests/test_onnx.py"),
    "onnxruntime": ("src/serving/", "tests/test_onnx.py"),
    "art": ("src/audit/",),
    "plotly": ("app/",),
}
# Facade aliases. Leading parameter names are checked; extra optional parameters are allowed.
FACADE = {
    "make_data": ["seed", "n"],
    "train_shards": ["df", "n_shards", "seed"],
    "predict_proba": ["X"],
    "shard_of": ["person_id"],
    "erase": ["person_ids", "request_id"],
    "audit": ["models", "members", "nonmembers"],
    "guard_check": ["batch"],
    "subgroup_delta": ["before", "after"],
    "make_receipt": [],
    "verify_chain": [],
}
NET_MODULES = {"requests", "urllib", "socket", "httpx", "http", "ftplib", "smtplib"}
CONST_NAMES = {"SEED", "N_SHARDS", "N_PATIENTS", "N_TEST", "N_CANARY", "THRESHOLDS"}
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", "demo_out"}


def _ann(value) -> str:
    if value is inspect.Signature.empty:
        return ""
    text = str(value).replace("'", "").replace('"', "")
    for prefix in ("pandas.", "pd.", "sklearn.ensemble.", "src.contracts.", "numpy.", "np."):
        text = text.replace(prefix, "")
    return text.replace(" ", "")


def _same_default(spec_default, real_default) -> bool:
    if spec_default is inspect.Parameter.empty and real_default is inspect.Parameter.empty:
        return True
    try:
        return spec_default == real_default
    except Exception:
        return repr(spec_default) == repr(real_default)


def _py_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def _check_constants(errors: list[str]) -> None:
    import src.contracts as c

    expected = {
        "SEED": 42,
        "N_SHARDS": 8,
        "N_PATIENTS": 20000,
        "N_TEST": 2000,
        "N_CANARY": 400,
    }
    for name, value in expected.items():
        got = getattr(c, name)
        if got != value:
            errors.append(f"contracts.{name} is {got!r}, spec requires {value!r}")
    features = [
        "age", "sex", "bmi", "sbp", "dbp", "hba1c", "smoker",
        "sym_fatigue", "sym_vision", "sym_thirst", "sym_chest_pain",
        "village_code", "prior_visits",
    ]
    if list(c.FEATURES) != features:
        errors.append(f"FEATURES mismatch: {list(c.FEATURES)}")
    thresholds = {
        "referral_threshold": 0.5,
        "n_bootstrap": 1000,
        "ci_level": 0.95,
        "n_shadow_models": 16,
        "fpr_target": 0.01,
        "n_bins": 10,
        "psi_moderate": 0.10,
        "psi_major": 0.25,
        "min_subgroup_size": 200,
        "confidence_drop": 0.10,
        "subgroup_auroc_drop": 0.03,
        "probe_rows": 1000,
        "batch_size": 500,
        "n_batches": 20,
        "drift_start_batch": 10,
        "drift_shift": 1.5,
        "drift_village": 3,
        "drift_feature": "hba1c",
    }
    for key, value in thresholds.items():
        if key not in c.THRESHOLDS:
            errors.append(f"THRESHOLDS missing key {key!r}")
        elif c.THRESHOLDS[key] != value:
            errors.append(f"THRESHOLDS[{key!r}] is {c.THRESHOLDS[key]!r}, spec requires {value!r}")
    fields = set(c.ShardSet.__dataclass_fields__)
    need = {"models", "n_shards", "models_dir", "manifest"}
    if not need <= fields:
        errors.append(f"ShardSet fields {sorted(fields)} missing {sorted(need - fields)}")
    leaked = [name for name in IMPLEMENTATIONS if name in c.__all__]
    if leaked:
        errors.append(f"contracts.__all__ exports function stubs: {leaked}")


def _check_signatures(names: list[str], errors: list[str], warnings: list[str]) -> None:
    import src.contracts as contracts

    for name in names:
        module_name = IMPLEMENTATIONS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"MISSING {module_name}.{name}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"IMPORT FAILED {module_name}.{name}: {type(exc).__name__}: {exc}")
            continue
        real = getattr(module, name, None)
        spec = getattr(contracts, name, None)
        if real is None or not callable(real):
            errors.append(f"{module_name} has no callable {name}")
            continue
        if spec is None:
            errors.append(f"contracts.py has no spec stub for {name}")
            continue
        real_params = list(inspect.signature(real).parameters.values())
        spec_params = list(inspect.signature(spec).parameters.values())
        if len(real_params) < len(spec_params):
            errors.append(
                f"{module_name}.{name} has fewer parameters than the spec: "
                f"real={[p.name for p in real_params]} spec={[p.name for p in spec_params]}"
            )
            continue
        for index, spec_param in enumerate(spec_params):
            real_param = real_params[index]
            if real_param.name != spec_param.name:
                errors.append(
                    f"{module_name}.{name} param {index} is {real_param.name!r}, "
                    f"spec requires {spec_param.name!r}"
                )
                continue
            if real_param.kind != spec_param.kind:
                errors.append(
                    f"{module_name}.{name} param {spec_param.name} kind "
                    f"{real_param.kind} != spec {spec_param.kind}"
                )
            if not _same_default(spec_param.default, real_param.default):
                errors.append(
                    f"{module_name}.{name} param {spec_param.name} default "
                    f"{real_param.default!r} != spec {spec_param.default!r}"
                )
        for extra in real_params[len(spec_params):]:
            if extra.default is inspect.Parameter.empty and extra.kind not in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
            ):
                errors.append(
                    f"{module_name}.{name} extra parameter {extra.name!r} has no default "
                    "(extra parameters must be optional)"
                )
        spec_ann = _ann(inspect.signature(spec).return_annotation)
        real_ann = _ann(inspect.signature(real).return_annotation)
        if spec_ann and real_ann and spec_ann != real_ann:
            warnings.append(
                f"{module_name}.{name} return annotation {real_ann!r} != spec {spec_ann!r}"
            )
        elif spec_ann and not real_ann:
            warnings.append(f"{module_name}.{name} is missing the spec return annotation {spec_ann}")


def _check_reexports(errors: list[str]) -> None:
    try:
        queue = importlib.import_module("src.erase.queue")
        receipt = importlib.import_module("src.report.receipt")
    except Exception as exc:
        errors.append(f"could not compare append_log re-export: {exc}")
        return
    if queue.append_log is not receipt.append_log:
        errors.append("src.erase.queue.append_log is not src.report.receipt.append_log")
    if queue.verify_log is not receipt.verify_log:
        errors.append("src.erase.queue.verify_log is not src.report.receipt.verify_log")


def _assigned_names(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append(node.target.id)
    return found


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_sources(errors: list[str], warnings: list[str]) -> None:
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    for path in _py_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "src/contracts.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            errors.append(f"{rel} does not parse: {exc}")
            continue
        if rel.startswith("src/") or rel.startswith("scripts/") or rel.startswith("bench/"):
            for name in _assigned_names(tree):
                if name in CONST_NAMES:
                    errors.append(f"{rel} assigns {name}; import it from src.contracts")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                errors.append(f"{rel}:{node.lineno} calls built-in hash()")
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif node.module and node.level == 0:
                modules = [node.module.split(".")[0]]
            for mod in modules:
                if mod in NET_MODULES and rel.startswith("src/"):
                    errors.append(f"{rel}:{node.lineno} imports network module {mod}")
                if mod in EXTRA_IMPORTS:
                    if not any(rel.startswith(prefix) or rel == prefix for prefix in EXTRA_IMPORTS[mod]):
                        errors.append(f"{rel}:{node.lineno} imports {mod} outside its allowed file")
                    continue
                if mod in stdlib or mod in ALLOWED_THIRD_PARTY or mod == "src":
                    continue
                if rel.startswith("tests/") and mod in {"pytest", "fakes_erase", "receipt_fixtures"}:
                    continue
                # local app package
                if rel.startswith("app/") and mod in {"app"}:
                    continue
                if mod not in stdlib and mod not in ALLOWED_THIRD_PARTY:
                    # ignore relative-less local modules that live in the same tree
                    local = list(ROOT.rglob(f"{mod}.py")) + list(ROOT.rglob(f"{mod}/__init__.py"))
                    if local:
                        continue
                    errors.append(f"{rel}:{node.lineno} imports disallowed package {mod}")
        if rel.startswith("src/") or rel.startswith("scripts/") or rel.startswith("app/"):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_getenv = (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"get", "getenv"}
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "environ"
                )
                if not is_getenv or len(node.args) < 2:
                    continue
                key = node.args[0]
                default = node.args[1]
                key_s = key.value if isinstance(key, ast.Constant) else ""
                if key_s == "FORGETME_HMAC_KEY" and not (
                    isinstance(default, ast.Constant) and default.value in (None, "")
                ):
                    errors.append(f"{rel}:{node.lineno} gives FORGETME_HMAC_KEY a default value")


def _check_receipt_whitelist(errors: list[str]) -> None:
    try:
        receipt = importlib.import_module("src.report.receipt")
    except Exception as exc:
        errors.append(f"receipt whitelist not checked: {exc}")
        return
    overlap = set(receipt.ALLOWED_KEYS) & set(receipt.PATIENT_FIELD_DENYLIST)
    if overlap:
        errors.append(f"receipt allows patient-level keys: {sorted(overlap)}")
    for key in ("person_id", "features", "needs_referral", "age", "hba1c"):
        if key in receipt.ALLOWED_KEYS:
            errors.append(f"receipt ALLOWED_KEYS contains {key}")


def _check_facade(errors: list[str]) -> None:
    try:
        facade = importlib.import_module("src.forgetme_api")
    except Exception as exc:
        errors.append(f"src.forgetme_api failed to import: {type(exc).__name__}: {exc}")
        return
    for name, expected in FACADE.items():
        fn = getattr(facade, name, None)
        if not callable(fn):
            errors.append(f"facade missing callable {name}")
            continue
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        leading = [p.name for p in params[:len(expected)]]
        if leading != expected:
            errors.append(f"facade {name} parameters start {leading}, expected {expected}")
        for extra in params[len(expected):]:
            if extra.default is inspect.Parameter.empty:
                errors.append(f"facade {name} extra parameter {extra.name} is required")
    if facade.shard_of is not importlib.import_module("src.model.shards").shard_of:
        errors.append("facade shard_of is not src.model.shards.shard_of")


def _check_known_traps(warnings: list[str]) -> None:
    queue = ROOT / "src" / "erase" / "queue.py"
    shards = ROOT / "src" / "model" / "shards.py"
    mia = ROOT / "src" / "audit" / "mia.py"
    bench = ROOT / "bench" / "run_benchmarks.py"
    if queue.exists() and shards.exists():
        q = queue.read_text(encoding="utf-8")
        s = shards.read_text(encoding="utf-8")
        if 'ent["hash"]' in q and '"model_hash"' in s:
            warnings.append(
                "erase manifest writer sets shards[k]['hash']; the shard layer stores shards[k]['model_hash']. "
                "After erase the manifest model_hash can stay stale."
            )
        if "train_current.csv" not in (bench.read_text(encoding="utf-8") if bench.exists() else ""):
            warnings.append(
                "bench/run_benchmarks.py does not mention train_current.csv. "
                "A full-retrain timing taken from train.csv after an erase is not the current training set."
            )
    if mia.exists():
        text = mia.read_text(encoding="utf-8")
        if "bootstrap_resamples" in text:
            warnings.append(
                "src/audit/mia.py reads THRESHOLDS['bootstrap_resamples']; "
                "contracts.py names that value n_bootstrap. The fallback is 1000, so the number matches."
            )
        if "gate_shard_auc" in text or "gate_ensemble_auc" in (ROOT / "scripts" / "audit_gate.py").read_text(encoding="utf-8"):
            warnings.append(
                "audit gate keys gate_shard_auc / gate_ensemble_auc are not in THRESHOLDS. "
                "Fallbacks are the frozen 0.90 and 0.62."
            )
    if not (ROOT / "src" / "guard" / "monitor.py").exists():
        warnings.append("src/guard/monitor.py is absent . Guard fields on a real erase stay null.")


# ------------------------------------------------------------ structure checks
# Patterns are assembled from pieces so this file never matches its own rules.
_DEV_PATH = re.compile("d" + "ev[ _-]?\\d", re.IGNORECASE)
_DEV_LABEL = re.compile("\\b" + "De" + "v\\s?\\d")
REQUIRED_FILES = (
    "README.md", "pyproject.toml", "requirements.txt", "requirements-optional.txt",
    "Makefile", ".gitignore", ".env.example",
    "data/.gitkeep", "models/.gitkeep", "logs/.gitkeep", "receipts/.gitkeep",
    "docs/architecture.md", "docs/ibm_status.md", "docs/s390x_risks.md",
    "docs/vm_runbook.md", "docs/local_env.md",
    "docs/project/STATUS.md", "docs/project/MERGE_LOG.md",
    "docs/project/SPEC_CHANGES.md", "docs/project/DEMO_RUNBOOK.md",
)
REQUIRED_DIRS = ("src", "app", "scripts", "bench", "tests", "docs/evidence")
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def _tracked_files() -> "list[str] | None":
    """Paths known to git (relative, posix), or None when this is not a git checkout."""
    if not (ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def _check_structure(errors: list[str]) -> None:
    tracked = _tracked_files()
    if tracked is None:
        tracked = []
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if path.is_file() and not any(part in SKIP_DIRS | {".git"} for part in rel.parts):
                tracked.append(rel.as_posix())
    for rel in tracked:
        parts = rel.split("/")
        if any(_DEV_PATH.search(part) for part in parts):
            errors.append(f"structure: dev-named path {rel}")
        if "__pycache__" in parts or rel.endswith(".pyc"):
            errors.append(f"structure: compiled cache is tracked: {rel}")
        f = ROOT / rel
        if f.is_file() and f.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"structure: {rel} is over 5 MB")
    for top in ("src", "app", "scripts", "bench", "tests"):
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if _DEV_LABEL.search(path.read_text(encoding="utf-8", errors="replace")):
                errors.append(f"structure: numbered-owner label in {rel}")
    for pkg in sorted(p for p in (ROOT / "src").rglob("*") if p.is_dir()):
        if any(part in SKIP_DIRS for part in pkg.parts):
            continue
        if any(pkg.glob("*.py")) and not (pkg / "__init__.py").is_file():
            errors.append(f"structure: package {pkg.relative_to(ROOT).as_posix()} has no __init__.py")
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            errors.append(f"structure: missing {rel}")
    for rel in REQUIRED_DIRS:
        if not (ROOT / rel).is_dir():
            errors.append(f"structure: missing directory {rel}/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check PRD 8.1 constants and 8.3 signatures")
    parser.add_argument("--wave", type=int, choices=(0, 1, 2, 3), default=3)
    args = parser.parse_args(argv)
    errors: list[str] = []
    warnings: list[str] = []
    _check_constants(errors)
    names = WAVE_FUNCTIONS[args.wave]
    if names:
        _check_signatures(names, errors, warnings)
    if args.wave >= 1 and "append_log" in names:
        _check_reexports(errors)
    _scan_sources(errors, warnings)
    if args.wave >= 1:
        _check_receipt_whitelist(errors)
        _check_facade(errors)
    _check_known_traps(warnings)
    _check_structure(errors)
    print(f"contract_check wave {args.wave}: {len(errors)} error(s), {len(warnings)} warning(s)")
    for item in errors:
        print(f"ERROR  {item}")
    for item in warnings:
        print(f"WARN   {item}")
    if errors:
        print("CONTRACT CHECK: FAIL")
        return 1
    print("CONTRACT CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
