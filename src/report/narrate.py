"""Plain-language erasure report (FR-13).

Three public pieces, all matching PRD 8.3:

  template_report(receipt)       deterministic wording, every figure taken from the receipt
  granite_report(receipt)        same output contract, written by a local Granite model
  validate_numbers(text, receipt) every number in the text must be traceable to the receipt

plus ``generate_report`` (try Granite, validate, fall back to the template) and
``generate_report_with_info`` which also returns what happened.

Design rules
  * Line 1 of every report is ``Writer: ...`` and says which writer produced it.
  * The validator skips that first line (it is added by code, not by a model).
  * A figure counts as "in the receipt" if it equals a receipt value after rounding to
    the number of decimals shown, or (when followed by %) equals the value times 100.
  * Hex identifiers (hash prefixes), the request id and the timestamp are checked as
    identifiers: they must occur verbatim inside the receipt, they are not "numbers".
  * Static allowlist (spec gap 7): legal / method constants that are not in a receipt.
  * No network calls anywhere in this file. Granite runs as a local subprocess.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from src.contracts import N_SHARDS, SEED, THRESHOLDS

REPO_ROOT = Path(__file__).resolve().parents[2]
GRANITE_LOG = REPO_ROOT / "docs" / "evidence" / "granite_attempt.log"

# --------------------------------------------------------------------------- allowlist
# Numbers that may legitimately appear in a report but are not receipt fields.
# Each entry says why. Keep this list short: every entry weakens the check.
STATIC_ALLOWLIST: dict[str, str] = {
    "2023": "year of the Digital Personal Data Protection Act",
    str(int(round(THRESHOLDS["ci_level"] * 100))): "confidence level of the attack-AUC interval (contracts)",
    str(int(round(THRESHOLDS["fpr_target"] * 100))): "1% false-positive rate used for TPR (contracts)",
    str(THRESHOLDS["n_shadow_models"]): "number of shadow models in the audit (contracts)",
    str(N_SHARDS): "number of shards in the deployed model (contracts)",
}

WRITER_PREFIX = "Writer:"
NOT_RECORDED = "not recorded"

LEGAL_NOTE = (
    "This is not legal advice. The legal status of trained model weights under the "
    "Digital Personal Data Protection Act, 2023 (DPDP Act) is unsettled, and the DPDP "
    "clause references and commencement dates mentioned here are to be verified with "
    "counsel before anyone relies on them."
)

# --------------------------------------------------------------------------- tiny helpers
_HEX_KEYS = ("hash", "hmac", "sha")


def _get(d, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _num(v) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _f(v, nd: int) -> str:
    """Fixed decimals, decimal half-up on the receipt's own decimal text (not the binary
    float), so 0.7775 always shows as 0.778 on every architecture."""
    if not _num(v):
        return NOT_RECORDED
    d = Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{d:.{nd}f}"


def _i(v) -> str:
    return str(int(v)) if _num(v) else NOT_RECORDED


def _short(h) -> str:
    return (str(h)[:12]) if isinstance(h, str) and h else NOT_RECORDED


def _plural(n, one: str, many: str) -> str:
    return one if _num(n) and int(n) == 1 else many


# --------------------------------------------------------------------------- template
def _audit_sentence(label: str, block: dict) -> str:
    """One sentence about an audit block. Wording is qualitative where possible."""
    if not isinstance(block, dict) or "attack_auc" not in block:
        return f"{label}: {NOT_RECORDED}."
    auc = block.get("attack_auc")
    ci = block.get("auc_ci95") or [None, None]
    tpr = block.get("tpr_at_1pct_fpr")
    pct = block.get("canary_percentile_median")
    level = f"{int(round(THRESHOLDS['ci_level'] * 100))}%"
    return (
        f"{label}: the attacker's score (attack AUC) was {_f(auc, 3)} "
        f"({level} interval {_f(ci[0], 3)} to {_f(ci[1], 3)}); "
        f"it correctly picked out {_f(Decimal(str(tpr)) * 100 if _num(tpr) else None, 1)}% of the audited "
        f"records at a 1% false-alarm rate; "
        f"the median canary percentile was {_f(pct, 1)}."
    )


def _verdict(before: dict, after: dict) -> str:
    """Qualitative reading of the audit. No digits. Empirical evidence, not proof."""
    try:
        b = float(before["attack_auc"])
        a = float(after["attack_auc"])
        lo, hi = after["auc_ci95"]
        lo, hi = float(lo), float(hi)
    except (KeyError, TypeError, ValueError):
        return ("The audit figures are incomplete in this receipt, so no reading of "
                "the before and after comparison is given.")
    if lo <= 0.5 <= hi:
        reading = ("After erasure the attacker does no better than a coin flip, within "
                   "the margin of error.")
    elif a < b:
        reading = ("After erasure the attacker did worse than before, but still better "
                   "than a coin flip, so some trace may remain.")
    else:
        reading = ("After erasure the attacker did not do worse than before. The audit "
                   "does not show that the erased records are forgotten.")
    return reading + (" The audit is empirical evidence from simulated attacks on this "
                      "model. It is not a mathematical proof of forgetting.")


def _guard_stream_sentence(g: dict) -> str:
    """Counts from the Guard stream run (receipt guard.drift_flags, guard.held_decisions).

    Counts only: the receipt never holds person ids. A decision "held for human review" waits
    for a person; the system never refuses anyone on its own.
    """
    flags, held = g.get("drift_flags"), g.get("held_decisions")
    if not _num(flags) and not _num(held):
        return "The Guard stream check (drift and held decisions) is " + NOT_RECORDED + " in this receipt."
    parts = []
    if _num(flags):
        parts.append(
            f"{_i(flags)} {_plural(flags, 'village', 'villages')} reached the major drift line"
        )
    else:
        parts.append("the number of villages at the major drift line is " + NOT_RECORDED)
    if _num(held):
        parts.append(
            f"{_i(held)} {_plural(held, 'decision was', 'decisions were')} held for human review"
        )
    else:
        parts.append("the number of decisions held for human review is " + NOT_RECORDED)
    return ("Guard stream check, counts only: " + ", and ".join(parts) + ". "
            "Held decisions wait for a person to look at them; nothing is refused automatically.")


def template_report(receipt: dict) -> str:
    """Deterministic plain-language report. Every figure is formatted from ``receipt``."""
    r = receipt or {}
    shard_ids = r.get("shard_ids") if isinstance(r.get("shard_ids"), list) else []
    n_sh = len(shard_ids)
    shard_txt = ", ".join(str(s) for s in shard_ids) if shard_ids else NOT_RECORDED
    mh_b, mh_a = r.get("model_hash_before") or {}, r.get("model_hash_after") or {}
    hash_lines = []
    for k in sorted(mh_b, key=lambda x: str(x)):
        changed = mh_a.get(k) is not None and mh_a.get(k) != mh_b.get(k)
        hash_lines.append(
            f"  Shard {k}: fingerprint before {_short(mh_b.get(k))}, after "
            f"{_short(mh_a.get(k))} ({'changed' if changed else 'unchanged'})."
        )
    ab, aa = r.get("audit_before") or {}, r.get("audit_after") or {}
    g = r.get("guard") if isinstance(r.get("guard"), dict) else None

    L: list[str] = []
    L.append(f"{WRITER_PREFIX} template (fixed wording, figures filled in from the receipt; "
             "no language model used)")
    L.append("")
    L.append("ERASURE REPORT")
    L.append(f"Request {r.get('request_id', NOT_RECORDED)}, recorded at "
             f"{r.get('timestamp_utc', NOT_RECORDED)}.")
    L.append("")
    L.append("Summary")
    L.append(
        f"{_i(r.get('persons_erased'))} {_plural(r.get('persons_erased'), 'person was', 'people were')} "
        f"erased from the triage model. {_i(r.get('rows_removed'))} "
        f"{_plural(r.get('rows_removed'), 'record was', 'records were')} removed from the training data. "
        f"Only {_plural(n_sh, 'the shard', 'the shards')} that held their records "
        f"({shard_txt}) {_plural(n_sh, 'was', 'were')} retrained; the other shards were left alone. "
        f"That is {_i(n_sh)} of {N_SHARDS} shards."
    )
    L.append("")
    L.append("Erasure (DPDP duty: erase personal data when it is no longer to be kept)")
    L.append(
        "The records were deleted from the training set and the affected shard was "
        "retrained from scratch without them, so the new model never saw them."
    )
    speed_line = (
        f"Retraining the affected shard took {_f(r.get('shard_retrain_seconds'), 2)} seconds. "
        f"Retraining every shard at once on the cores available to this process would have taken "
        f"{_f(r.get('full_retrain_seconds'), 2)} seconds, so the shard retrain was "
        f"{_f(r.get('speedup'), 1)} times faster than that parallel baseline."
    )
    if r.get("full_retrain_serial_seconds") is not None:
        speed_line += (
            f" A one-at-a-time retrain of every shard measured "
            f"{_f(r.get('full_retrain_serial_seconds'), 2)} seconds"
        )
        if r.get("speedup_vs_serial") is not None:
            speed_line += (
                f", which is {_f(r.get('speedup_vs_serial'), 1)} times the shard retrain."
            )
        else:
            speed_line += "."
    L.append(speed_line)
    L.append("")
    L.append("Consent withdrawal (DPDP duty: act on a withdrawal of consent)")
    L.append(
        f"The withdrawal was handled as request {r.get('request_id', NOT_RECORDED)} and "
        f"written to the erasure log as entry number {_i(r.get('log_seq'))}. "
        "Erasing the data from the model is how this clinic acts on the withdrawal."
    )
    L.append("")
    L.append("Integrity (DPDP duty: reasonable security safeguards)")
    L.extend(hash_lines or ["  Model fingerprints: " + NOT_RECORDED + "."])
    L.append(
        f"The log entry fingerprint starts with {_short(r.get('log_entry_hash'))}. Each log "
        "entry includes the fingerprint of the one before it, so editing an old entry "
        "breaks the chain. This receipt carries a signature (HMAC) made with a secret key. "
        "This report does not check that signature; the verification step does."
    )
    L.append("")
    L.append("Did the model forget? (membership-inference audit)")
    L.append(_audit_sentence("Whole model, before", ab.get("ensemble")))
    L.append(_audit_sentence("Whole model, after", aa.get("ensemble")))
    L.append(_audit_sentence("Affected shard, before", ab.get("shard")))
    L.append(_audit_sentence("Affected shard, after", aa.get("shard")))
    L.append(_verdict(ab.get("shard") or {}, aa.get("shard") or {}))
    L.append("")
    L.append("Did the model stay fair and accurate?")
    L.append(
        f"Accuracy on the held-out test set (AUROC) was {_f(r.get('test_auroc_before'), 3)} "
        f"before and {_f(r.get('test_auroc_after'), 3)} after."
    )
    if g is None:
        L.append("The safety guard check is " + NOT_RECORDED + " in this receipt.")
    else:
        L.append(
            f"Largest drop in accuracy for any village or age band: "
            f"{_f(g.get('max_subgroup_auroc_drop'), 3)}. Worst-affected group: "
            f"{g.get('worst_subgroup') if g.get('worst_subgroup') is not None else NOT_RECORDED}."
        )
        L.append(_guard_stream_sentence(g))
        flagged = g.get("flagged_subgroups")
        if isinstance(flagged, dict) and flagged:
            bits = [f"{k} {_f(v, 3)}" for k, v in sorted(flagged.items())]
            L.append("Subgroup drops listed on the receipt: " + "; ".join(bits) + ".")
        elif isinstance(flagged, dict):
            L.append("No subgroup drop is listed on this receipt.")
    L.append("")
    L.append("Limits")
    L.append(
        "The data in this system is synthetic. The audit tests specific attacks; a "
        "different attack could do better. " + LEGAL_NOTE
    )
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- validator
_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\b")
_REQ_RE = re.compile(r"\bREQ-\d+\b")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def _dec(s) -> Decimal | None:
    try:
        d = Decimal(str(s))
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _collect(obj, nums: set, strings: list, key: str = "") -> None:
    """Gather numeric values (absolute) and string leaves from the receipt."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        d = _dec(obj)
        if d is not None:
            nums.add(abs(d))
    elif isinstance(obj, str):
        strings.append(obj)
        if any(h in key.lower() for h in _HEX_KEYS) or _HEX_RE.fullmatch(obj):
            return  # hashes are identifiers, not numbers
        for m in _NUM_RE.finditer(obj):
            d = _dec(m.group(1).replace(",", ""))
            if d is not None:
                nums.add(abs(d))
    elif isinstance(obj, dict):
        nums.add(Decimal(len(obj)))
        for k, v in obj.items():
            _collect(str(k), nums, strings, "")
            _collect(v, nums, strings, str(k))
    elif isinstance(obj, (list, tuple)):
        nums.add(Decimal(len(obj)))  # "1 shard" is derived from len(shard_ids)
        for v in obj:
            _collect(v, nums, strings, key)


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def _equal_at(cand: Decimal, shown: Decimal, nd: int) -> bool:
    """``shown`` is ``cand`` rounded to nd decimals, in any rounding mode: it is within
    half a unit of the last shown digit (inclusive, so exact ties match either way)."""
    half = Decimal(5).scaleb(-nd - 1)
    return abs(cand - shown) <= half


def _number_matches(tok: str, percent: bool, corpus: set) -> bool:
    shown = _dec(tok.replace(",", ""))
    if shown is None:
        return False
    nd = _decimals(tok.replace(",", ""))
    for v in corpus:
        cands = [v] + ([v * 100] if percent else [])
        for c in cands:
            if nd == 0 and not percent:
                if c == shown:          # integers must match exactly (no rounding to 0)
                    return True
            elif _equal_at(c, shown, nd):
                return True
    return False


def validate_numbers(text: str, receipt: dict) -> tuple[bool, list[str]]:
    """Check that every number in ``text`` is traceable to ``receipt``.

    Returns (ok, unmatched) where ``unmatched`` lists the offending numbers or
    identifiers exactly as written, in order of first appearance.
    A leading ``Writer: ...`` line is skipped (it is added by code, not by a writer).
    """
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith(WRITER_PREFIX):
        lines = lines[1:]
    body = "\n".join(lines)

    nums: set = set()
    strings: list = []
    _collect(receipt, nums, strings)
    for s in STATIC_ALLOWLIST:
        d = _dec(s)
        if d is not None:
            nums.add(d)
    hay = "\n".join(strings).lower()
    bad: list[str] = []

    def add(x: str) -> None:
        if x not in bad:
            bad.append(x)

    def eat(pattern, check_in_receipt: bool, keep_if_missing_digits: bool = False):
        nonlocal body
        def sub(m):
            tok = m.group(0)
            if not check_in_receipt or tok.lower() in hay:
                return " "
            if keep_if_missing_digits and tok.isdigit():
                return tok                     # plain digits: let the numeric check judge
            add(tok)
            return " "
        body = pattern.sub(sub, body)

    eat(_ISO_RE, True)
    eat(_REQ_RE, True)
    eat(_HEX_RE, True, keep_if_missing_digits=True)

    for m in _NUM_RE.finditer(body):
        tok = m.group(1)
        percent = body[m.end(): m.end() + 1] == "%"
        if not _number_matches(tok, percent, nums):
            add(tok)
    return (not bad, bad)


# --------------------------------------------------------------------------- granite
class GraniteUnavailable(RuntimeError):
    """Granite is not configured or not runnable on this machine."""


PROMPT = """You write a short, plain-language erasure report for a rural clinic manager who is not technical.
Use ONLY the numbers that appear in the receipt JSON below. Do not compute new numbers, do not
round differently than needed, do not add any number that is not in the receipt. If a value is
missing or null, write "not recorded".
Use these headings: Summary, Erasure, Consent withdrawal, Integrity, Did the model forget?,
Did the model stay fair and accurate?, Limits.
Map erasure, consent withdrawal and integrity to the DPDP Act duties in plain words.
In Limits say: not legal advice; the legal status of model weights under DPDP is unsettled;
DPDP dates to be verified. Do not claim the audit is a proof. Do not use numbered lists.

RECEIPT JSON:
{receipt_json}

REPORT:
"""


def build_prompt(receipt: dict) -> str:
    """The prompt contains only the instructions and the receipt JSON (no patient data exists in it)."""
    return PROMPT.format(receipt_json=json.dumps(receipt, sort_keys=True, indent=1))


def _run_llama(prompt: str) -> tuple[str, str]:
    """Run llama.cpp locally. Returns (text, model_file_name). Raises GraniteUnavailable."""
    cli = os.environ.get("FORGETME_LLAMA_CLI", "")
    model = os.environ.get("FORGETME_GRANITE_GGUF", "")
    if not cli or not model:
        raise GraniteUnavailable(
            "not configured (set FORGETME_LLAMA_CLI and FORGETME_GRANITE_GGUF)")
    if not os.path.isfile(cli) or not os.access(cli, os.X_OK):
        raise GraniteUnavailable(f"llama.cpp binary not executable: {cli}")
    if not os.path.isfile(model):
        raise GraniteUnavailable(f"model file not found: {model}")
    extra = shlex.split(os.environ.get("FORGETME_LLAMA_ARGS", "-no-cnv --no-display-prompt"))
    n_tok = os.environ.get("FORGETME_LLAMA_NTOK", "700")
    timeout = int(os.environ.get("FORGETME_LLAMA_TIMEOUT", "900"))
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "prompt.txt")
        with open(pf, "w", encoding="utf-8") as fh:
            fh.write(prompt)
        cmd = [cli, "-m", model, "-f", pf, "-n", n_tok, "--temp", "0",
               "--seed", str(SEED)] + extra
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               check=False)
        except subprocess.TimeoutExpired as e:
            raise GraniteUnavailable(f"llama.cpp timed out after {timeout}s") from e
        except OSError as e:
            raise GraniteUnavailable(f"could not start llama.cpp: {e}") from e
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-1:] or [""]
        raise GraniteUnavailable(f"llama.cpp exited with code {p.returncode}: {tail[0][:160]}")
    out = p.stdout.replace("[end of text]", "").strip()
    if not out:
        raise GraniteUnavailable("llama.cpp produced no text")
    return out, os.path.basename(model)


def granite_report(receipt: dict) -> str:
    """Report written by a local Granite model. Same output contract as template_report:
    a string whose first line starts with ``Writer:``. Raises GraniteUnavailable on any
    runtime problem. Does NOT validate numbers; ``generate_report`` does that.
    Status: Planned until it has run on the VM (see docs/ibm_status.md)."""
    text, model_name = _run_llama(build_prompt(receipt))
    first = f"{WRITER_PREFIX} IBM Granite via llama.cpp (local subprocess; model file {model_name})"
    return first + "\n\n" + text + "\n"


# --------------------------------------------------------------------------- orchestration
def _log(path, status: str, detail: str) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(f"{ts} | {status} | {detail.replace(chr(10), ' ')[:400]}\n")
    except OSError:
        pass  # logging must never break report generation


def generate_report_with_info(receipt: dict, log_path=GRANITE_LOG) -> tuple[str, dict]:
    """Try Granite; validate its numbers; fall back to the template and say so.

    Returns (text, info) with info = {writer: 'granite'|'template', fallback_reason, unmatched}.
    """
    reason, unmatched = None, []
    try:
        text = granite_report(receipt)
        ok, unmatched = validate_numbers(text, receipt)
        if ok:
            _log(log_path, "OK", "granite report passed number validation")
            return text, {"writer": "granite", "fallback_reason": None, "unmatched": []}
        reason = "Granite output rejected, numbers not in receipt: " + ", ".join(unmatched)
        _log(log_path, "REJECTED", reason)
    except GraniteUnavailable as e:
        reason = f"Granite not used: {e}"
        _log(log_path, "UNAVAILABLE", str(e))
    except Exception as e:  # any other runtime error also falls back
        reason = f"Granite failed ({type(e).__name__}: {e})"
        _log(log_path, "ERROR", reason)
    body = template_report(receipt).split("\n", 1)[1]
    first = (f"{WRITER_PREFIX} template (fallback: {reason}; fixed wording, figures filled "
             "in from the receipt)").replace("\n", " ")
    return first + "\n" + body, {"writer": "template", "fallback_reason": reason,
                                 "unmatched": unmatched}


def generate_report(receipt: dict) -> str:
    return generate_report_with_info(receipt)[0]


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m src.report.narrate RECEIPT.json", file=__import__('sys').stderr)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        rec = json.load(fh)
    text, info = generate_report_with_info(rec)
    ok, bad = validate_numbers(text, rec)
    print(text)
    print(f"[validator] writer={info['writer']} ok={ok} unmatched={bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(__import__("sys").argv))
