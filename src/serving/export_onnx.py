"""ONNX export of the shard models (FR-17, export half).

skl2onnx / onnx / onnxruntime are the only extra packages allowed in the project and
only here. They are imported lazily: if they will not install on s390x within the
time-box, keep the install log and report this item as "Planned".

STATUS: written but NOT yet run (skl2onnx unavailable in the authoring sandbox).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.contracts import FEATURES, MODELS_DIR, ShardSet

ONNX_DIR_NAME = "onnx"
PARITY_TOL = 1e-5
PARITY_ROWS = 1000


def _need_deps():
    try:
        import onnxruntime  # noqa: F401
        import skl2onnx  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ONNX export needs skl2onnx + onnxruntime (not installed). Report the "
            "ONNX item as 'Planned' and keep the install log.") from exc


def export_all(ss: ShardSet, out_dir: str | None = None) -> list[str]:
    """Write models/onnx/shard_k.onnx for every shard. Returns the paths."""
    _need_deps()
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    out_dir = out_dir or os.path.join(ss.models_dir or MODELS_DIR, ONNX_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, model in sorted(ss.models.items()):
        onx = convert_sklearn(
            model, initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
            options={id(model): {"zipmap": False}})
        path = os.path.join(out_dir, f"shard_{k}.onnx")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(onx.SerializeToString())
        os.replace(tmp, path)
        paths.append(path)
    return paths


def onnx_predict_shard(path: str, X: pd.DataFrame) -> np.ndarray:
    """P(class 1) from one .onnx shard."""
    _need_deps()
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X[FEATURES].to_numpy(dtype=np.float32)})
    return np.asarray(out[1])[:, 1].astype(np.float64)  # outputs: label, probabilities


def parity_report(ss: ShardSet, onnx_dir: str, X: pd.DataFrame) -> dict:
    """Max abs difference sklearn vs onnxruntime, per shard and for the ensemble."""
    from src.model.shards import predict
    X = X.head(PARITY_ROWS)
    sk = predict(ss, X)
    per_shard, ens = {}, []
    for k in sorted(ss.models):
        p = onnx_predict_shard(os.path.join(onnx_dir, f"shard_{k}.onnx"), X)
        per_shard[k] = float(np.max(np.abs(p - sk["per_shard"][k])))
        ens.append(p)
    ens_p = np.mean(np.vstack(ens), axis=0)
    return {"rows": len(X), "tolerance": PARITY_TOL,
            "max_diff_per_shard": per_shard,
            "max_diff_ensemble": float(np.max(np.abs(ens_p - sk["proba"]))),
            "max_diff_overall": max(max(per_shard.values()),
                                    float(np.max(np.abs(ens_p - sk["proba"]))))}


if __name__ == "__main__":
    from src.model.shards import load, load_probe
    ss_ = load()
    export_all(ss_)
    rep = parity_report(ss_, os.path.join(ss_.models_dir, ONNX_DIR_NAME), load_probe())
    print(rep)
    print("PASS" if rep["max_diff_overall"] <= PARITY_TOL else
          "FAIL: report the max diff to the team; do not loosen the tolerance")
