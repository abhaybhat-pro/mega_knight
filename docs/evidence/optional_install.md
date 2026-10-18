# Optional packages (WSL2 x86_64 rehearsal)

Attempted 2026-10-17 in a throwaway venv at `/tmp/forgetme-optional`, not the product `.venv`. Time box 12 minutes. `uv` 0.13.1. The product environment was not modified.

| Package | Result |
| --- | --- |
| diffprivlib | Installed, then `ImportError: cannot import name 'DOUBLE' from 'sklearn.tree._tree'`. Label: Planned |
| skl2onnx | `IMPORT_OK` in the throwaway venv only |
| onnx | `IMPORT_OK` in the throwaway venv only |
| onnxruntime | `IMPORT_OK` in the throwaway venv only |

Gate 3 on the product venv: `1 passed, 2 skipped` (`tests/test_dp.py`, `tests/test_onnx.py` skip when the import is missing; `tests/test_shards_curve.py` passed). ONNX and differential privacy stay Planned. Nothing here was run on s390x.
