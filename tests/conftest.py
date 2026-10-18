"""Pin BLAS to one thread before any test module imports NumPy; skip linux_only tests on Windows."""
import sys

import pytest

from src.runtime.resources import configure_threads

configure_threads()


def pytest_collection_modifyitems(config, items):
    """Tests marked linux_only need fork, /proc or a POSIX shell: skip them cleanly elsewhere."""
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(reason="linux_only: needs fork, /proc or a POSIX shell")
    for item in items:
        if "linux_only" in item.keywords:
            item.add_marker(skip)
