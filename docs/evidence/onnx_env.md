# ONNX export attempt (one attempt, dedicated venv, x86_64 cloud container, not s390x)

Venv outside the repo: `uv venv --python 3.12`, then numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, skl2onnx 1.20.0, onnx 1.23.2, onnxruntime 1.31.0.
Install and test log: `docs/evidence/onnx_attempt_x86_64.log` (`tests/test_onnx.py`: 1 passed).
Parity numbers: `docs/evidence/onnx_parity_x86_64.txt`. Largest absolute probability difference across the 8 shards and 1,000 probe rows: 2.9e-07 (frozen tolerance 1e-5); ensemble 1.9e-07.

Scope: export of the 8 shard forests to ONNX and parity with scikit-learn on x86_64. It says nothing about s390x, the IBM Z Deep Learning Compiler or Triton, which stay Planned.
