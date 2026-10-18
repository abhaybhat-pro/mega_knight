#!/usr/bin/env python3
"""V8/V9 helper: repository hygiene and documentation consistency. Exit 1 on any problem.

Checks: secret patterns in the working tree and the full git history; tracked files over 5 MB;
tracked caches/bytecode/archives/models/data; dev-named paths; .gitignore covers generated data;
every relative markdown link in README and docs resolves; README layout tree matches the repo.
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout


problems = []
tracked = [f for f in git("ls-files").splitlines() if f]

# 1. tracked junk and size
for f in tracked:
    p = ROOT / f
    if p.exists() and p.stat().st_size > 5 * 1024 * 1024:
        problems.append(f"tracked file over 5 MB: {f}")
    if re.search(r"(__pycache__|\.pyc$|\.pytest_cache|\.zip$|\.gguf$|\.onnx$|\.joblib$|\.key$|(^|/)\.env$)", f):
        problems.append(f"junk or secret-like file tracked: {f}")
    if re.match(r"(data|models|logs|receipts)/", f) and not f.endswith(".gitkeep"):
        problems.append(f"generated runtime file tracked: {f}")
    if re.search(r"(?i)(^|/)dev[ _-]?\d", f):
        problems.append(f"dev-named path: {f}")

# 2. secrets in the tree and in all history
PAT = re.compile(r"(?i)(FORGETME_HMAC_KEY\s*=\s*['\"]?(?!\$|throwaway|demo|vm-|your|<)[A-Za-z0-9+/_\-]{12,}"
                 r"|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
                 r"|xox[bap]-[A-Za-z0-9-]{10,})")
hits = []
for f in tracked:
    p = ROOT / f
    if p.is_file() and p.stat().st_size < 2_000_000:
        try:
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if PAT.search(line):
                    hits.append(f"{f}:{n}")
        except OSError:
            pass
hist = subprocess.run(["git", "log", "-p", "--all", "--no-color"], cwd=ROOT, capture_output=True, text=True).stdout
for line in hist.splitlines():
    if line.startswith("+") and PAT.search(line):
        hits.append("history: " + line[:80])
for h in sorted(set(hits)):
    problems.append(f"possible secret: {h}")
gi = (ROOT / ".gitignore").read_text()
for need in ("data/*", "models/*", "logs/*", "receipts/*", ".venv/", "__pycache__/", ".env", "*.key"):
    if need not in gi:
        problems.append(f".gitignore does not cover {need}")

# 3. links
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
docs = [ROOT / "README.md"] + [p for p in (ROOT / "docs").rglob("*.md") if "evidence" not in p.parts]
for d in docs:
    for m in LINK.finditer(d.read_text(encoding="utf-8")):
        tgt = m.group(1)
        if re.match(r"[a-z]+:", tgt):
            continue
        if not (d.parent / tgt).exists():
            problems.append(f"broken link in {d.relative_to(ROOT)}: {tgt}")

# 4. tree
r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_tree.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
if r.returncode:
    problems.append("README tree does not match the repository: " + (r.stdout + r.stderr).strip()[:200])

print(f"check_hygiene: {len(problems)} problem(s) ({len(tracked)} tracked files)")
print("\n".join(problems))
sys.exit(1 if problems else 0)
