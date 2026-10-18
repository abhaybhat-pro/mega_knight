#!/usr/bin/env python3
"""Fail if any tracked .py file uses syntax newer than Python 3.10 (the oldest interpreter the VM image may ship)."""
import ast, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
bad = []
files = [f for f in ROOT.rglob("*.py") if not {".venv", "__pycache__", ".git"} & set(f.parts)]
for f in files:
    try:
        ast.parse(f.read_text(encoding="utf-8"), feature_version=(3, 10))
    except SyntaxError as e:
        bad.append(f"{f.relative_to(ROOT)}: {e}")
print(f"{len(files)} files checked against the Python 3.10 grammar; {len(bad)} problem(s)")
print("\n".join(bad))
sys.exit(1 if bad else 0)
