"""ONNX export parity (FR-17 export half). Skips when skl2onnx/onnxruntime are absent
(then report the item as Planned). Tolerance 1e-5 is FROZEN; never loosen it."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

try:
    import onnxruntime  # noqa: F401
    import skl2onnx  # noqa: F401
except ImportError:
    pytest.skip("skl2onnx/onnxruntime not installed: ONNX item is Planned",
                allow_module_level=True)

from src.contracts import TRAIN_CSV  # noqa: E402
from src.data.generate import generate, read  # noqa: E402
from src.model import shards  # noqa: E402
from src.serving import export_onnx as ex  # noqa: E402


def test_onnx_matches_sklearn_within_1e5():
    d = tempfile.mkdtemp()
    generate(out_dir=d)
    probe = shards.load_probe(d)
    ss = shards.train_all(read(TRAIN_CSV, d), n_workers=1,
                          models_dir=tempfile.mkdtemp(), probe_X=probe)
    onnx_dir = os.path.join(ss.models_dir, "onnx")
    paths = ex.export_all(ss, onnx_dir)
    assert len(paths) == 8 and all(os.path.exists(p) for p in paths)
    rep = ex.parity_report(ss, onnx_dir, probe)
    assert rep["rows"] == 1000
    assert rep["max_diff_overall"] <= 1e-5, rep  # report the real number if this fails
