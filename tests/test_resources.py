"""plan_workers, thread pinning, and the resource monitor. Utilization is not asserted."""
import os
import sys

import pytest

from src.runtime.resources import (
    REPORT_KEYS,
    ResourceMonitor,
    available_cores,
    configure_threads,
    plan_workers,
)


def test_configure_threads_sets_blas_env(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    configure_threads()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[name] == "1"


def test_plan_workers_honours_cores_tasks_and_env(monkeypatch):
    cores = available_cores()
    assert plan_workers(1, 1.0) == 1
    wide = plan_workers(10_000, 1.0)
    assert 1 <= wide <= cores
    monkeypatch.setenv("FORGETME_MAX_WORKERS", "1")
    assert plan_workers(10_000, 1.0) == 1
    monkeypatch.setenv("FORGETME_MAX_WORKERS", "2")
    monkeypatch.setenv("FORGETME_MEM_BUDGET_MB", "100")
    assert plan_workers(10_000, 80.0) == 1
    monkeypatch.setenv("FORGETME_MEM_BUDGET_MB", "500")
    # The env cap can never create cores that do not exist: on a 1-core machine the
    # answer is 1 even with FORGETME_MAX_WORKERS=2.
    assert plan_workers(10_000, 80.0) == min(2, cores)


@pytest.mark.parametrize("cores,cap,expected", [
    (1, "2", 1),    # cap above the real core count: bounded by the cores
    (2, "2", 2),
    (8, "2", 2),    # cap below the core count: the cap wins
    (8, "", 8),     # no cap: all cores
    (3, "8", 3),    # cap above cores on a 3-core VM
])
def test_plan_workers_cap_never_exceeds_cores(monkeypatch, cores, cap, expected):
    """Regression guard for the env-cap logic, independent of this machine's core count."""
    import src.runtime.resources as res
    monkeypatch.setattr(res, "available_cores", lambda: cores)
    monkeypatch.setattr(res, "mem_available_mb", lambda: 16000.0)
    monkeypatch.setattr(res, "mem_total_mb", lambda: 32000.0)
    monkeypatch.delenv("FORGETME_MEM_BUDGET_MB", raising=False)
    monkeypatch.setenv("FORGETME_MAX_WORKERS", cap)
    assert plan_workers(10_000, 80.0) == expected


@pytest.mark.linux_only
def test_resource_monitor_keys_on_linux():
    with ResourceMonitor(interval=0.05) as mon:
        sum(range(1000))
    report = mon.report()
    assert set(REPORT_KEYS) <= set(report)
    assert report["samples"] >= 1
    assert report["peak_rss_mb"] is None or report["peak_rss_mb"] >= 0
