#!/usr/bin/env python3
"""Honesty check over the prose a reader or a judge sees (README, docs, UI text).

Fails (exit 1) on:
  - "mathematically prove(s)"
  - deny / denied / auto-deny / reject(ed) used as a system outcome (the Guard only proceeds, warns or holds for review)
  - "exactly 0.5" (post-erase AUC is described as "at or below chance")
  - "false flag(s)" (use "advisories")
  - "8x" on a line that does not say it is against a serial retrain
  - Telum, Spyre or Secure Execution on a line that does not say it is not used / not claimed
  - "Dev N" owner labels
  - an evidence path cited in the README or docs that does not exist
Lines inside fenced "negative examples" can opt out with the marker  <!-- claims:ok -->.
Run from anywhere:  python scripts/check_claims.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", "evidence", "history"}
SKIP_FILES = {"SPEC_CHANGES.md", "MERGE_LOG.md"}          # change records quote the old wording on purpose
TEXT = ("*.md", "*.txt")

RULES = [
    ("proof wording", re.compile(r"mathematically\s+prove", re.I), None),
    ("refusal outcome", re.compile(r"\b(auto-?deny|denied|deny|denies|rejected|reject)\b", re.I), None),
    ("exact 0.5", re.compile(r"exactly\s+0\.5\b", re.I), None),
    ("false flags", re.compile(r"false[- ]flags?\b", re.I), None),
    ("owner label", re.compile(r"\bDev ?[0-9]\b"), None),
    ("8x without serial", re.compile(r"\b8(\.\d+)?x\b", re.I), re.compile(r"serial", re.I)),
    ("hardware claim", re.compile(r"\b(Telum|Spyre|Secure Execution)\b", re.I),
     re.compile(r"not (used|claimed|benchmarked)|never claim|no claim|do(es)? not|is not|are not", re.I)),
]
EVID = re.compile(r"docs/evidence/[\w./\-]+")


def files():
    for pat in TEXT:
        for f in ROOT.rglob(pat):
            if SKIP_DIRS & set(f.relative_to(ROOT).parts) or f.name in SKIP_FILES:
                continue
            yield f
    for f in (ROOT / "app").rglob("*.py"):                  # UI strings
        yield f


def main() -> int:
    problems: list[str] = []
    for f in sorted(set(files())):
        rel = f.relative_to(ROOT)
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "claims:ok" in line:
                continue
            for name, rx, ok in RULES:
                if rx.search(line) and not (ok and ok.search(line)):
                    problems.append(f"{rel}:{n}: {name}: {line.strip()[:110]}")
            if f.suffix == ".md":
                for m in EVID.finditer(line):
                    path = m.group(0).rstrip(".,)`:;")
                    if "*" in path or "<" in path or path.endswith(("_", "-")):
                        continue
                    if not (ROOT / path).exists():
                        problems.append(f"{rel}:{n}: evidence path does not exist: {path}")
    print(f"check_claims: {len(problems)} problem(s)")
    print("\n".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
