# FR-17 serve half: ZDLC + Triton (PLANNED, nothing here has been run)

Status: **Planned**. This is a checklist of what to try, not evidence. Every command below
is unverified and must be checked against IBM's current documentation before use.

## Inputs
The `.onnx` files from `src/serving/export_onnx.py` (needs `skl2onnx` + `onnxruntime`;
that install is itself Planned on s390x, see `requirements.txt`).

## Steps to attempt, in order, stopping at the first blocker
1. Check what the VM allows: `docker --version` or `podman --version`; can it pull from an
   IBM Z container registry; is there any outbound network. Record the output in
   `triton_attempt.log`.
2. Obtain the IBM Z Deep Learning Compiler (ZDLC) image or build, per IBM's docs.
3. Compile one shard: `.onnx` -> shared library with the ZDLC. Record the exact command, the
   tool version and whether it accepts the tree-ensemble operators skl2onnx produces
   (ONNX-ML `TreeEnsembleClassifier`). If the operator is unsupported, that is the finding.
4. Serve with Triton Inference Server for IBM Z, using the compiled model. Record the command.
5. Measure latency with a small client against the served model: median of repeated calls,
   with the VM core count. Save to `docs/triton_latency.csv`.

## What to write down either way
`docs/evidence/triton_attempt.log`: command, UTC time, exit code, last error line. If blocked, mark
the row "Planned" in `ibm_status.md` and keep the log: the evidence of the attempt is the
deliverable. Do not describe serving as working unless step 5 produced numbers.
