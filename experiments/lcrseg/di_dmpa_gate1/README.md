# Gate 1A only: execution notes

This package is an offline diagnostic, not a registered method. It never
imports the training runner or constructs a segmentation/transport optimizer.
The two preregistration files remain immutable. Authorization is the separate
25ec97c988af290a4fb7a637c4b7cdfe462deb87 commit.

Execution barrier: commit/push all source and verify `git ls-remote` before
the real-checkpoint integration check or formal run. Formal `run` checks the
exact remote code commit, preregistration/authorization ancestry and bytes,
config hashes, all checkpoints and input roles. It refuses an existing output
directory. It then builds and locks the shared sampling plan, writes complete
run metadata (with the actual plan hash), starts two independent GPU feature
shards, verifies every cache and model state, and only then runs CPU float64
clustering/bootstraps and final all-panel adjudication.

The setup metadata describes the input-audit/plan-building phase, not a model
diagnostic shard. No feature worker starts with an unknown sampling-plan hash.
Validation GT is read only by coordinate/morphology preparation and evaluation;
it cannot enter a fit task's training array. No test or train_unlabeled record
can be constructed through this package's Gate 1A role interface.

## Tests and environment

Use the existing remote `/root/.venvs/lcrseg-py310/bin/python`; do not install
or replace Torch. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Run from `experiments/lcrseg` with
`PYTHONPATH=.`. The official classifier constructor resets RNG, so each
source/role forward is reseeded after loading with its registered geometry
pixel-sampling seed. Classifier stochasticity is disabled throughout.

Development runs select `test_gate1a_core.py` and the optional real integration
file only. Do not rerun the old prework test that updates synthetic optimizers
as part of this no-optimizer Gate 1A suite.

For the committed-source integration test, explicitly set
`GATE1A_CODE_COMMIT` and `GATE1A_REAL_CHECKPOINT_REPORT`. It reads the first
registered B0 stage0 checkpoint and first registered labeled case at eight
constant image coordinates, on both source models. It reads no label arrays,
fits no prototypes and produces no geometry/admission metric.

The formal CLI needs a directory with passing exact-code test evidence:
`GATE1A_UNIT_INTEGRATION_TEST_REPORT.json`, `pytest.xml`, `pytest_output.txt`,
and `REAL_CHECKPOINT_EXTRACTION_INTEGRATION.json`. The report must carry
`status=PASS` and the exact `diagnostic_code_git_commit`.

```text
python -m di_dmpa_gate1.gate1a_runner run \
  --code-commit FULL_PUSHED_CODE_SHA \
  --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/UNIQUE_ATTEMPT \
  --tests ABSOLUTE_TEST_EVIDENCE_DIR --gpus 0,1 --workers 16 --planning-workers 8
```

No mode exists for transport, reliability, gradient conflict, teacher-noise,
theory, training, Gate 2 or main merge. GPU shard scheduling is absent from
all sampling/clustering seeds. Aggregation never stops controls after seeing
a primary performance result; any protocol/numerical failure blocks instead.

## Preserved prepublication development failures

The first synthetic suite at `/root/LCRSeg/runs/gate1a_unit_dev_attempt1`
reported 44 passes and one source-scan failure: macOS tar copied AppleDouble
`._*.py` metadata into the development sandbox. No formal data/model was
read. Source transfer is corrected with `COPYFILE_DISABLE=1`, `--no-xattrs`
and exclusion of `._*`, into a new sandbox; the failed sandbox and transcript
remain untouched. This run also recorded CUDA error 804 from the container
compatibility library path; use the existing working system-library path
above, without changing drivers or packages. An earlier rsync transfer could
not start because rsync is absent remotely; tar transport is used instead.
