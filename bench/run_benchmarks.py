"""Scaling benchmarks (FR-10).

Writes bench/results.csv with exactly these columns:
    experiment, n_workers, n_shards, seconds, n_cores
``seconds`` is the MEDIAN of ``--repeats`` timings (default 3). ``n_cores`` is
``available_cores()``. Only measured values are written. Also writes
bench/results_meta.json (which training file was timed) and bench/scaling_notes.md.
``--profile`` writes bench/resource_profile.csv. The audit stand-in is gone (SC-6):
audit rows call ``run_audit(..., n_workers=)``.

Experiments
  erase_vs_full  seconds to retrain ONLY the affected shard after removing a member-canary
                 batch (n_workers is 1: a single shard is a single task). Compare with the
                 train_scaling rows at the same n_shards for the "retrain everything" cost.
  train_scaling  seconds to train all shards from scratch in a process pool.
                 n_shards = N_SHARDS for every worker count, plus the 1/4/16-shard
                 sweep at n_workers = 1 (the sequential baseline for each shard count).
  audit_scaling  seconds for src.audit.mia.run_audit with n_workers passed through, measured
                 COLD: every timed run gets a fresh, empty models dir, so the 128 shadow
                 forests are built inside the timed region. Only worker counts up to the
                 core count are timed, and the median is over --audit-repeats (default 1,
                 because one cold build already takes minutes on few cores). A warm-cache
                 audit is only scoring and is NOT a scaling result: with --warm-audit the
                 rows are written as audit_warm_cache so they cannot be mistaken for it.
  full_retrain_parallel / full_retrain_serial
                 fit_models on the current training file (train_current.csv when it
                 exists, else train.csv). The parallel row is the fair clinic baseline.

Run:  python bench/run_benchmarks.py            (full, needs data/train.csv)
      python bench/run_benchmarks.py --quick    (tiny smoke run, writes bench/_tmp/)
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.runtime.resources import (  # noqa: E402
    ResourceMonitor, available_cores, configure_threads, wave_ceiling,
)

configure_threads()

import pandas as pd  # noqa: E402

from src.contracts import (CANARY_COL, DATA_DIR, HELDOUT_CSV, ID_COL, MODELS_DIR,  # noqa: E402
                           N_SHARDS, SEED, STREAM_CSV, TEST_CSV, THRESHOLDS, TRAIN_CSV,
                           ShardSet)
from src.model.shards import assign_shards, fit_models, train_shard  # noqa: E402

CSV_COLUMNS = ["experiment", "n_workers", "n_shards", "seconds", "n_cores"]
EXPERIMENTS = (
    "erase_vs_full", "train_scaling", "audit_scaling",
    "full_retrain_parallel", "full_retrain_serial",
)
PROFILE_COLUMNS = [
    "phase", "n_workers", "wall_s", "avg_cpu_pct", "peak_cpu_pct", "peak_rss_mb",
    "min_mem_avail_mb", "disk_used_mb", "host_label",
]
HOST_LABEL = "x86_64 rehearsal, not s390x"
SHARD_SWEEP = (1, 4, 16)            # PRD 8.1: "benchmarks also run 1, 4, 16"
BASE_WORKERS = (1, 2, 4, 8)         # block spec; extended up to the VM core count


# ------------------------------------------------------------------ environment
def get_nproc() -> int:
    """Cores this process can run on. Same number ``available_cores`` reports."""
    return available_cores()


def worker_counts(n_cores: int) -> list[int]:
    """1, 2, 4, 8 always; then doubling and finally n_cores itself, up to the VM core count."""
    ws = set(BASE_WORKERS) | {n_cores}
    w = 16
    while w <= n_cores:
        ws.add(w)
        w *= 2
    return sorted(ws)


def median_seconds(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ------------------------------------------------------------------ workloads
def _real_audit_callable():
    """Return (callable|None, note). Real only if run_audit takes an n_workers argument."""
    try:
        from src.audit import mia  # type: ignore
        sig = inspect.signature(mia.run_audit)
    except Exception as e:  # not built yet, or cannot import
        return None, f"src.audit.mia.run_audit unavailable ({type(e).__name__}); stand-in used"
    if "n_workers" not in sig.parameters:
        return None, ("src.audit.mia.run_audit has no n_workers parameter; stand-in used "
                      "(add an optional n_workers argument to run_audit)")
    return mia.run_audit, "real src.audit.mia.run_audit"


def _target_shard(n_shards: int) -> int:
    return 3 % n_shards  # same shard as the PRD receipt example


def _erase_workload(train_df: pd.DataFrame, n_shards: int):
    """Remove a member-canary batch (all member canaries in the target shard) and retrain
    only that shard. Row removal is inside the timed region, as it is in a real erase."""
    sid = assign_shards(train_df[ID_COL], n_shards)
    k = _target_shard(n_shards)
    canaries = train_df.loc[(sid == k) & (train_df[CANARY_COL] == "member"), ID_COL]
    if len(canaries) == 0:  # tiny smoke run: drop a few rows instead
        canaries = train_df.loc[sid == k, ID_COL].head(5)
    ids = set(canaries.tolist())

    def run():
        remaining = train_df[~train_df[ID_COL].isin(ids)]
        shard_rows = remaining.loc[assign_shards(remaining[ID_COL], n_shards) == k]
        train_shard(shard_rows, k, SEED)

    return run, len(ids)


# ------------------------------------------------------------------ driver
def run_benchmarks(train_df: pd.DataFrame, out_dir: str, repeats: int = 3,
                   workers: list[int] | None = None, shard_counts=SHARD_SWEEP,
                   experiments=EXPERIMENTS, quick: bool = False, log=print,
                   heldout_df: pd.DataFrame | None = None,
                   training_file: str = TRAIN_CSV,
                   data_dir: str = DATA_DIR,
                   audit_repeats: int = 1, cold_audit: bool = True) -> list[dict]:
    n_cores = get_nproc()
    workers = workers or worker_counts(n_cores)
    rows: list[dict] = []
    meta: dict = {"n_cores": n_cores, "repeats": repeats, "workers": workers,
                  "n_train_rows": int(len(train_df)), "training_file": training_file,
                  "quick": quick,
                  "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "python": sys.version.split()[0], "byteorder": sys.byteorder,
                  "machine": os.uname().machine, "experiments": {}}

    def add(exp, w, n, secs):
        rows.append({"experiment": exp, "n_workers": w, "n_shards": n,
                     "seconds": round(secs, 4), "n_cores": n_cores})
        log(f"  {exp:14s} workers={w:<3d} shards={n:<3d} median={secs:8.3f}s")

    if "train_scaling" in experiments:
        meta["experiments"]["train_scaling"] = {
            "source": "src.model.shards.fit_models, no disk I/O",
            "n_shards_main": N_SHARDS}
        pairs = [(N_SHARDS, w) for w in workers] + [(n, 1) for n in shard_counts if n != N_SHARDS]
        for n, w in pairs:
            add("train_scaling", w, n,
                median_seconds(lambda n=n, w=w: fit_models(train_df, n, w), repeats))

    if "erase_vs_full" in experiments:
        meta["experiments"]["erase_vs_full"] = {
            "source": "remove member-canary batch + src.model.shards.train_shard on the one "
                      "affected shard. NOT the end-to-end erase() (audit, log, receipt not timed)",
            "n_workers": 1}
        for n in sorted(set([N_SHARDS, *shard_counts])):
            run, n_ids = _erase_workload(train_df, n)
            add("erase_vs_full", 1, n, median_seconds(run, repeats))
            meta["experiments"]["erase_vs_full"].setdefault("ids_removed", {})[str(n)] = n_ids

    if "audit_scaling" in experiments:
        real, note = _real_audit_callable()
        meta["experiments"]["audit_scaling"] = {
            "source": note, "is_real": real is not None and heldout_df is not None,
            "standin": None, "cold_cache": bool(cold_audit),
            "repeats": int(audit_repeats),
            "method": ("each timed run uses a fresh empty models dir, so the shadow cache is "
                       "built inside the timed region" if cold_audit else
                       "WARM shadow cache: this times scoring only and is not a scaling result")}
        if real is not None and heldout_df is not None:
            models = fit_models(train_df, N_SHARDS, -1)  # setup, not timed
            members = train_df[train_df[CANARY_COL] == "member"]
            audit_workers = [w for w in workers if w <= n_cores] or [1]
            meta["experiments"]["audit_scaling"].update(
                n_members=int(len(members)), n_nonmembers=int(len(heldout_df)),
                workers_timed=audit_workers,
                workers_not_timed=("above the core count: a CPU-bound shadow build cannot scale "
                                   "past the cores, so those counts were not timed"))

            def audit_once(w):
                if cold_audit:
                    with tempfile.TemporaryDirectory(prefix="bench_audit_cold_") as tmp:
                        ss = ShardSet(models=models, n_shards=N_SHARDS, models_dir=tmp)
                        real(ss, members, heldout_df, n_workers=w, data_dir=data_dir)
                else:
                    ss = ShardSet(models=models, n_shards=N_SHARDS, models_dir=MODELS_DIR)
                    real(ss, members, heldout_df, n_workers=w, data_dir=data_dir)

            name = "audit_scaling" if cold_audit else "audit_warm_cache"
            for w in audit_workers:
                add(name, w, N_SHARDS, median_seconds(lambda w=w: audit_once(w), audit_repeats))
        else:
            log("  audit_scaling skipped: run_audit or held-out canaries are missing")

    if "full_retrain_parallel" in experiments:
        meta["experiments"]["full_retrain_parallel"] = {
            "source": "fit_models with n_workers=-1 (plan_workers on the measured cores)",
            "headline": "full_retrain_seconds"}
        add("full_retrain_parallel", -1, N_SHARDS,
            median_seconds(lambda: fit_models(train_df, N_SHARDS, -1), repeats))
    if "full_retrain_serial" in experiments:
        meta["experiments"]["full_retrain_serial"] = {
            "source": "fit_models with n_workers=1, labelled separately from the parallel baseline"}
        add("full_retrain_serial", 1, N_SHARDS,
            median_seconds(lambda: fit_models(train_df, N_SHARDS, 1), repeats))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "results_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "scaling_notes.md"), "w", encoding="utf-8") as fh:
        fh.write(scaling_notes(rows, meta))
    return rows


# ------------------------------------------------------------------ Amdahl note
def amdahl_serial_fraction(t_by_w: dict[int, float], n_cores: int) -> float | None:
    """Estimate serial fraction f from S(w) = 1 / (f + (1-f)/w), using only w > 1 and
    w <= n_cores (more workers than cores measures oversubscription, not scaling)."""
    t1 = t_by_w.get(1)
    if not t1:
        return None
    fs = []
    for w, t in t_by_w.items():
        if 1 < w <= n_cores and t > 0:
            s = t1 / t
            fs.append((1.0 / s - 1.0 / w) / (1.0 - 1.0 / w))
    if not fs:
        return None
    return min(1.0, max(0.0, statistics.median(fs)))


def scaling_notes(rows: list[dict], meta: dict) -> str:
    n_cores = meta["n_cores"]
    L = ["# Where scaling flattens (Amdahl)", "",
         f"Measured on a machine with {n_cores} core(s) (`nproc`), {meta['machine']}, "
         f"Python {meta['python']}, byte order {meta['byteorder']}, "
         f"{meta['n_train_rows']} training rows, median of {meta['repeats']} runs. "
         + ("QUICK SMOKE RUN: numbers are not benchmark results. " if meta["quick"] else "")
         + "Everything below is computed from bench/results.csv.", ""]
    for exp in ("train_scaling", "audit_scaling"):
        sub = [r for r in rows if r["experiment"] == exp and r["n_shards"] == N_SHARDS]
        if not sub:
            continue
        t = {r["n_workers"]: r["seconds"] for r in sub}
        L.append(f"## {exp} ({N_SHARDS} shards)")
        if exp == "audit_scaling":
            a = meta["experiments"]["audit_scaling"]
            L.append(f"Source: {a['source']}. Cold cache: {a.get('cold_cache')}. "
                     f"Repeats per row: {a.get('repeats')}. Worker counts above the core count were not timed.")
        L += ["", "| workers | seconds | speedup vs 1 worker |", "|---|---|---|"]
        for w in sorted(t):
            sp = t[1] / t[w] if 1 in t and t[w] > 0 else float("nan")
            L.append(f"| {w} | {t[w]:.3f} | {sp:.2f}x |")
        f = amdahl_serial_fraction(t, n_cores)
        L.append("")
        if n_cores == 1:
            L.append("This machine has one core, so no parallel speedup can be measured; rows "
                     "with more than one worker show process-pool overhead and oversubscription only.")
        elif f is None:
            L.append("Not enough rows with 1 < workers <= cores to estimate a serial fraction.")
        else:
            L.append(f"Estimated serial fraction (Amdahl, workers up to {n_cores}): "
                     f"{f:.3f}, which caps speedup at about "
                     f"{(1 / f if f > 0 else float('inf')):.1f}x.")
            within = sorted(w for w in t if w <= n_cores)
            flat = None
            for a, b in zip(within, within[1:]):
                if t[a] > 0 and (t[a] / t[b]) < 1.10:  # <10% further gain from doubling
                    flat = b
                    break
            if flat is not None:
                L.append(f"Scaling flattens at {flat} workers: the step from the previous "
                         "worker count gained less than 10%.")
        over = [w for w in t if w > N_SHARDS]
        if over:
            L.append(f"Workers above {N_SHARDS} cannot help train_scaling: there are only "
                     f"{N_SHARDS} shard tasks.")
        ceiling = wave_ceiling(N_SHARDS, n_cores)
        L.append(
            f"Eight shard tasks on {n_cores} cores fill "
            f"{ceiling:.0%} of the core-slots (the last wave can leave a core idle). "
            "That ceiling is the train_all limit on this machine. "
            f"{HOST_LABEL}."
        )
        L.append("")
    L += ["Causes to check (not measured here): process start-up and data transfer to workers, "
          "the slowest single shard (a pool cannot finish before it), memory bandwidth, and "
          "the cores of the VM being shared.", ""]
    return "\n".join(L)


def _host_label() -> str:
    try:
        machine = os.uname().machine
    except AttributeError:
        machine = ""
    if machine == "s390x":
        return "s390x measured on this host"
    return HOST_LABEL


def _timed_row(phase: str, n_workers: int, fn) -> dict:
    mon = ResourceMonitor(disk_path=ROOT)
    t0 = time.perf_counter()
    with mon:
        fn()
    wall = time.perf_counter() - t0
    rep = mon.report()
    return {
        "phase": phase,
        "n_workers": n_workers,
        "wall_s": round(wall, 4),
        "avg_cpu_pct": None if rep["avg_cpu_pct"] is None else round(rep["avg_cpu_pct"], 2),
        "peak_cpu_pct": None if rep["peak_cpu_pct"] is None else round(rep["peak_cpu_pct"], 2),
        "peak_rss_mb": None if rep["peak_rss_mb"] is None else round(rep["peak_rss_mb"], 1),
        "min_mem_avail_mb": None if rep["min_mem_avail_mb"] is None else round(rep["min_mem_avail_mb"], 1),
        "disk_used_mb": None if rep["disk_used_mb"] is None else round(rep["disk_used_mb"], 1),
        "host_label": _host_label(),
    }


def write_resource_profile(train_df: pd.DataFrame, heldout_df: pd.DataFrame | None,
                           data_dir: str, out_dir: str, log=print) -> str:
    """One measured pass of each phase. Utilization is printed, not asserted."""
    from src.audit.mia import get_shadows, run_audit
    from src.data.generate import drift_stream, generate
    from src.guard.monitor import fit_baseline, score_batch
    from src.model.shards import train_all
    from src.report.narrate import template_report
    from src.runtime.resources import estimate_forest_mb, plan_workers

    rows_n = max(1, len(train_df) // N_SHARDS)
    parallel_workers = plan_workers(N_SHARDS, estimate_forest_mb(rows_n))
    cores = available_cores()
    rows = []

    def add(phase, n_workers, fn):
        log(f"  profile {phase}")
        row = _timed_row(phase, n_workers, fn)
        rows.append(row)
        avg = row["avg_cpu_pct"]
        if n_workers > 1 and avg is not None:
            gate = "PASS" if avg >= 85.0 else "WARN"
            cause = ""
            if gate == "WARN":
                cause = (
                    f" likely cause: {n_workers} workers on {cores} cores, "
                    f"wave ceiling {wave_ceiling(N_SHARDS, cores):.0%}, "
                    "or the memory budget reduced the pool"
                )
            log(f"    soft gate {gate}: avg CPU {avg:.1f}% (line is 85% of the machine){cause}")
        return row

    gen_dir = os.path.join(out_dir, "_profile_generate")
    add("generate", 1, lambda: generate(out_dir=gen_dir))
    ss_box = {}
    # Fresh models dir: the shadow cache starts empty, so the shadow_build row is a COLD build
    # (a populated models/ would turn it into a ~0 s cache load) and models/ is left untouched.
    prof_models = tempfile.mkdtemp(prefix="bench_profile_models_")
    add("train_all", parallel_workers,
        lambda: ss_box.setdefault("ss", train_all(train_df, models_dir=prof_models)))
    ss = ss_box.get("ss")
    if ss is not None:
        add("shadow_build", plan_workers(128, estimate_forest_mb(rows_n)),
            lambda: get_shadows(ss, data_dir))
        if heldout_df is not None:
            members = train_df[train_df[CANARY_COL] == "member"]
            add("audit", plan_workers(128, estimate_forest_mb(rows_n)),
                lambda: run_audit(ss, members, heldout_df, data_dir=data_dir))
        sid = assign_shards(train_df[ID_COL], N_SHARDS)
        part = train_df.loc[sid == 3]
        add("erase_one_shard", 1, lambda: train_shard(part, 3, SEED))
    add("full_retrain_parallel", parallel_workers, lambda: fit_models(train_df, N_SHARDS, -1))
    add("full_retrain_serial", 1, lambda: fit_models(train_df, N_SHARDS, 1))
    if ss is not None and os.path.exists(os.path.join(data_dir, STREAM_CSV)):
        stream = pd.read_csv(os.path.join(data_dir, STREAM_CSV))
        test_path = os.path.join(data_dir, TEST_CSV)
        test = pd.read_csv(test_path) if os.path.exists(test_path) else None
        base = fit_baseline(train_df, ss=ss, test_df=test)

        def _guard():
            for batch in drift_stream(stream):
                score_batch(ss, batch, base)

        add("guard_stream", 1, _guard)
    add("report", 1, lambda: template_report({"request_id": "profile", "persons_erased": 0}))
    path = os.path.join(out_dir, "resource_profile.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=PROFILE_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    log(f"  wrote {path}")
    log(f"  Amdahl ceiling for {N_SHARDS} shard tasks on {cores} cores: "
        f"{wave_ceiling(N_SHARDS, cores):.0%}. {_host_label()}.")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="output dir (default bench/, or bench/_tmp with --quick)")
    ap.add_argument("--workers", default=None, help="comma list, default 1,2,4,8 + up to nproc")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (subsampled rows)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--profile", action="store_true",
                    help="write resource_profile.csv (one pass per phase)")
    ap.add_argument("--audit-repeats", type=int, default=1,
                    help="repeats for the COLD audit_scaling rows (default 1; each is a full shadow build)")
    ap.add_argument("--warm-audit", action="store_true",
                    help="time the audit against a warm shadow cache (written as audit_warm_cache, not scaling)")
    a = ap.parse_args(argv)

    from src.erase.queue import load_train_df

    current = os.path.join(a.data_dir, "train_current.csv")
    train_path = os.path.join(a.data_dir, TRAIN_CSV)
    if os.path.exists(current):
        which = "train_current.csv"
    elif os.path.exists(train_path):
        which = TRAIN_CSV
    else:
        print(f"{train_path} missing: run `python -m src.data.generate` first", file=sys.stderr)
        return 2
    df = load_train_df(a.data_dir)
    hpath = os.path.join(a.data_dir, HELDOUT_CSV)
    held = pd.read_csv(hpath) if os.path.exists(hpath) else None
    out = a.out or os.path.join(ROOT, "bench", "_tmp" if a.quick else "")
    workers = [int(x) for x in a.workers.split(",")] if a.workers else None
    shard_counts = SHARD_SWEEP
    if a.quick:
        df = df.sample(n=min(1600, len(df)), random_state=SEED).sort_values(ID_COL)
        shard_counts = (1, 4)
        workers = workers or [1, 2]
    exps = tuple(e for e in a.experiments.split(",") if e)
    bad = [e for e in exps if e not in EXPERIMENTS]
    if bad:
        print(f"unknown experiments {bad}", file=sys.stderr)
        return 2
    print(f"cores={get_nproc()}  rows={len(df)}  file={which}  repeats={a.repeats}  {_host_label()}",
          file=sys.stderr)
    run_benchmarks(df, out, a.repeats, workers, shard_counts, exps, a.quick,
                   log=lambda s: print(s, file=sys.stderr), heldout_df=held,
                   training_file=which, data_dir=a.data_dir,
                   audit_repeats=a.audit_repeats, cold_audit=not a.warm_audit)
    if a.profile:
        write_resource_profile(df, held, a.data_dir, out, log=lambda s: print(s, file=sys.stderr))
    print(os.path.join(out, "results.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
