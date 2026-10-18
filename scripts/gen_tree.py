#!/usr/bin/env python3
"""Print the repository layout from the files git tracks (or, outside git, from the disk).

    python scripts/gen_tree.py            print the tree
    python scripts/gen_tree.py --write    replace the block between the TREE markers in README.md
    python scripts/gen_tree.py --check    exit 1 if README.md's block differs from the real tree

docs/evidence/ is collapsed to one line (no count, so adding a log does not stale the README) because it holds dozens of logs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- TREE:BEGIN -->", "<!-- TREE:END -->"
COLLAPSE = ("docs/evidence/",)
SKIP_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache"}


def tracked_files() -> list[str]:
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        files = [p for p in out.split("\0") if p]
    else:
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()]
    return sorted(p for p in set(files)
                  if not (set(p.split("/")) & SKIP_PARTS) and (ROOT / p).exists())


def build_tree(files: list[str]) -> str:
    collapsed = {prefix: 0 for prefix in COLLAPSE}
    kept = []
    for rel in files:
        hit = next((c for c in COLLAPSE if rel.startswith(c)), None)
        if hit:
            collapsed[hit] += 1
        else:
            kept.append(rel)
    tree: dict = {}
    for rel in kept:
        node = tree
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None
    for prefix, count in collapsed.items():
        parts = prefix.strip("/").split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1] + "/  (logs and benchmark snapshots)"] = None
    lines = ["forgetme-aegis/"]

    def walk(node: dict, prefix: str) -> None:
        names = sorted(node, key=lambda n: ("/" not in n, n.lower()))
        for i, name in enumerate(names):
            last = i == len(names) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}")
            if isinstance(node[name], dict):
                walk(node[name], prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(lines)


def block() -> str:
    return f"{BEGIN}\n```text\n{build_tree(tracked_files())}\n```\n{END}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    new = block()
    readme = ROOT / "README.md"
    if not (args.write or args.check):
        print(new)
        return 0
    text = readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("README.md has no TREE markers", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    updated = head + new + tail
    if args.write:
        readme.write_text(updated, encoding="utf-8")
        print("README tree written")
        return 0
    if updated != text:
        print("README tree is out of date: run python scripts/gen_tree.py --write", file=sys.stderr)
        return 1
    print("README tree matches the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
