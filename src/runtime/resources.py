"""Detect cores, memory and disk, and size process pools. Stdlib only.

Used by every entry point.

``n_workers=-1`` means ``plan_workers``. Forests stay ``n_jobs=1``; the only
parallelism is the process pool, so BLAS is pinned to 1 thread in each process.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - Windows without the resource module
    resource = None

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# Keep this much RAM free of the pool, and stop adding workers below this.
_MEM_RESERVE_MB = 1500.0
_ARTIFACT_CAP_BYTES = 20 * 1024 ** 3
REPORT_KEYS = (
    "avg_cpu_pct",
    "peak_cpu_pct",
    "peak_rss_mb",
    "min_mem_avail_mb",
    "disk_used_mb",
    "samples",
)


def configure_threads() -> None:
    """Pin numeric libraries to one thread. Call before NumPy is imported."""
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def available_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _meminfo() -> dict[str, float]:
    path = Path("/proc/meminfo")
    out: dict[str, float] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if not parts:
            continue
        try:
            kib = float(parts[0])
        except ValueError:
            continue
        out[key] = kib / 1024.0  # /proc/meminfo reports kB
    return out


def mem_total_mb() -> float | None:
    return _meminfo().get("MemTotal")


def mem_available_mb() -> float | None:
    info = _meminfo()
    if "MemAvailable" in info:
        return info["MemAvailable"]
    if "MemFree" in info:
        return info["MemFree"]
    return None


def disk_free_mb(path: str | os.PathLike[str] = ".") -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024.0 ** 2)
    except OSError:
        return None


def disk_used_mb(path: str | os.PathLike[str] = ".") -> float | None:
    try:
        return shutil.disk_usage(path).used / (1024.0 ** 2)
    except OSError:
        return None


def estimate_forest_mb(n_rows: int) -> float:
    """Rough resident size of one RandomForest on ``n_rows`` rows.

    Used only to keep the pool inside RAM. It is not a benchmark result.
    """
    return max(100.0, 0.08 * max(1, int(n_rows)))


def plan_workers(n_tasks: int, est_task_mem_mb: float) -> int:
    """How many pool workers to start for ``n_tasks`` pieces of work.

    ``min(cores, n_tasks, floor(mem_budget / est_task_mem_mb))``.
    ``FORGETME_MAX_WORKERS`` caps the core count.
    ``FORGETME_MEM_BUDGET_MB`` replaces the computed memory budget.
    Below 1500 MB available the pool drops to one worker.
    """
    tasks = max(1, int(n_tasks))
    cores = available_cores()
    raw_cap = os.environ.get("FORGETME_MAX_WORKERS", "").strip()
    if raw_cap:
        cores = max(1, min(cores, int(raw_cap)))
    avail = mem_available_mb()
    total = mem_total_mb()
    if avail is not None and avail < _MEM_RESERVE_MB:
        return 1
    if avail is None or total is None:
        budget = None
    else:
        budget = min(0.8 * avail, total - _MEM_RESERVE_MB)
    raw_budget = os.environ.get("FORGETME_MEM_BUDGET_MB", "").strip()
    if raw_budget:
        budget = float(raw_budget)
    if budget is None or est_task_mem_mb <= 0:
        by_mem = tasks
    else:
        by_mem = max(1, int(budget // float(est_task_mem_mb)))
    return max(1, min(cores, tasks, by_mem))


def resolve_workers(n_workers: int, n_tasks: int, est_task_mem_mb: float) -> int:
    """``n_workers < 0`` (the project-wide default) asks ``plan_workers``."""
    if n_workers is None or int(n_workers) < 0:
        return plan_workers(n_tasks, est_task_mem_mb)
    return max(1, min(int(n_workers), max(1, int(n_tasks))))


def wave_ceiling(n_tasks: int, n_cores: int) -> float:
    """Fraction of core-slots filled when ``n_tasks`` run in waves of ``n_cores``.

    Eight shard tasks on three cores take three waves (3, 3, 2), so the ceiling
    is 8/9, about 89 percent. The last wave leaves one core idle.
    """
    cores = max(1, int(n_cores))
    tasks = max(0, int(n_tasks))
    if tasks == 0:
        return 0.0
    waves = (tasks + cores - 1) // cores
    return tasks / float(waves * cores)


def artifact_bytes(roots: list[str | os.PathLike[str]]) -> int:
    total = 0
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
            continue
        for dirpath, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    continue
    return total


def assert_artifact_budget(extra_bytes: int, roots: list[str | os.PathLike[str]]) -> None:
    """Refuse a write that would push generated artifacts over 20 GB."""
    used = artifact_bytes(roots)
    if used + max(0, int(extra_bytes)) > _ARTIFACT_CAP_BYTES:
        raise RuntimeError(
            f"artifact cap {_ARTIFACT_CAP_BYTES} bytes would be exceeded "
            f"(already {used}, adding {extra_bytes})"
        )


def _cpu_jiffies() -> tuple[int, int] | None:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except OSError:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu" or len(parts) < 5:
        return None
    nums = [int(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    return total, idle


def _rss_mb() -> float | None:
    if resource is None:
        return None
    try:
        self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (OSError, ValueError, AttributeError):
        return None
    # Linux reports ru_maxrss in kilobytes. Sum of the two peaks, as specified.
    return (self_kb + child_kb) / 1024.0


class ResourceMonitor:
    """Sample /proc/stat and /proc/meminfo about every 0.5 s.

    Fields are None where /proc is missing (a Windows host, for example).
    """

    def __init__(self, interval: float = 0.5, disk_path: str | os.PathLike[str] = "."):
        self.interval = interval
        self.disk_path = disk_path
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        cpu = _cpu_jiffies()
        self.samples.append({
            "cpu": cpu,
            "mem_avail": mem_available_mb(),
            "t": time.perf_counter(),
        })

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "ResourceMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def report(self) -> dict:
        pcts: list[float] = []
        prev = None
        mems: list[float] = []
        for sample in self.samples:
            if sample["mem_avail"] is not None:
                mems.append(float(sample["mem_avail"]))
            cur = sample["cpu"]
            if prev is not None and cur is not None:
                dt = cur[0] - prev[0]
                di = cur[1] - prev[1]
                if dt > 0:
                    pcts.append(100.0 * (dt - di) / dt)
            if cur is not None:
                prev = cur
        return {
            "avg_cpu_pct": (sum(pcts) / len(pcts)) if pcts else None,
            "peak_cpu_pct": max(pcts) if pcts else None,
            "peak_rss_mb": _rss_mb(),
            "min_mem_avail_mb": min(mems) if mems else None,
            "disk_used_mb": disk_used_mb(self.disk_path),
            "samples": len(self.samples),
        }
