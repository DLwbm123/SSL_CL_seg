# Gate 1A v2 exact commands and publication sequence

This records the single authorized v2 attempt. It is not authorization to rerun it, start v1 attempt3, or proceed downstream. All run directories are create-only; the launchers refuse an existing attempt.

## Published identities, in order

Branch: `codex/gate1a-v2-null-aware-sphere`, based on `606a5c53a37d0e4c9605415e8b38a1f177d1604f`, not main.

| Publication | Exact commit |
| --- | --- |
| v1 closure | `b61f6db0ca9e746d005937e7dfc51c45078e1d80` |
| v2 preregistration | `eaae37bbaa7546679d9e6893023afbeeef0ab5c6` |
| v2 execution authorization | `e8f558dcc3fb6054a3f757c1295bd07ede2a002b` |
| exact diagnostic code | `8ae5d7532f90aee5d53c0d966706ef64c18a19ac` |

Each was independently committed, pushed, and remotely verified before the next phase. In particular, preregistration and authorization preceded all new checkpoint tensor reads/forwards; exact code publication preceded the known-zero real integration test. No method implementation or training configuration was registered.

The remote identity check used:

```sh
git ls-remote https://github.com/DLwbm123/SSL_CL_seg.git refs/heads/codex/gate1a-v2-null-aware-sphere
git ls-remote https://github.com/DLwbm123/SSL_CL_seg.git refs/heads/main
```

The cloud checkout was installed from the same locally verified Git objects using an incremental bundle, then detached at the exact code SHA:

```sh
git -C /root/SSL_CL_gate1 fetch /root/code.bundle HEAD
git -C /root/SSL_CL_gate1 worktree add --detach /root/SSL_CL_gate1_v2 8ae5d7532f90aee5d53c0d966706ef64c18a19ac
mkdir -p /root/SSL_CL_gate1_v2/experiments/lcrseg/third_party
ln -s /root/SSL_CL_gate0_v2/experiments/lcrseg/third_party/JASCL_REFERENCE /root/SSL_CL_gate1_v2/experiments/lcrseg/third_party/JASCL_REFERENCE
git -C /root/SSL_CL_gate1_v2 status --porcelain
```

## Exact execution environment

Working directory: `/root/SSL_CL_gate1_v2/experiments/lcrseg`.

Existing interpreter: `/root/.venvs/lcrseg-py310/bin/python`. No package installation or environment replacement was performed.

```sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=.
```

The archived `launch_v2.py` supplies these exact variables, redirects output into create-only logs, detaches only the wrapper, and records PID, command, source SHA, launcher SHA and completion exit code. It never automatically advances tests to formal execution.

## Exact-code tests and real integration

Launch command:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 31192 root@162.14.139.38 '/root/.venvs/lcrseg-py310/bin/python /root/launch_v2.py tests'
```

Test-only additional environment:

```sh
export CUDA_VISIBLE_DEVICES=0
export GATE1A_CODE_COMMIT=8ae5d7532f90aee5d53c0d966706ef64c18a19ac
export GATE1A_V2_INTEGRATION_OUTPUT=/root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6/integration_8ae5d75_attempt1
```

Actual pytest command from the launcher:

```sh
/root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/di_dmpa_gate1_v2 tests/di_dmpa_gate1/test_gate1a_core.py tests/di_dmpa_gate1/test_gate1a_recovery.py --junitxml=/root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6/tests_8ae5d75_attempt1/pytest.xml
```

The real test uses the original eight-case batch containing `REFUGE_test_n0128`, pixel `(125,212)`, class1, B0 seed2 stage0 EMA. It does not fit prototypes or run clustering. The fixed unit-center distance assertion is a test fixture, not a diagnostic fit.

## Unique formal v2 attempt

The tests were explicitly inspected and confirmed PASS before this separate launch:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 31192 root@162.14.139.38 '/root/.venvs/lcrseg-py310/bin/python /root/launch_v2.py formal'
```

Actual formal command:

```sh
/root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1_v2.runner run --code-commit 8ae5d7532f90aee5d53c0d966706ef64c18a19ac --output /root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6/gate1a_v2_8ae5d7532f90aee5d53c0d966706ef64c18a19ac_attempt1 --tests /root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6/tests_8ae5d75_attempt1 --gpus 0,1 --workers 16
```

Each feature shard receives `CUDA_VISIBLE_DEVICES=0` or `1`. Fixed batch size8, float32 forward, no AMP, eval/no_grad, stochastic classifier disabled. Geometry uses 16 CPU workers with single-threaded BLAS and float64. The original plan is copied byte-for-byte, not regenerated; neither v1 partial cache is loaded.

## Read-only postrun verification and report publication

The archived `postrun_v2_audit.py` hashes every formal artifact, both historical attempt manifests and their files, and all18 original checkpoints. It summarizes already-written JSON fit warnings, checks coverage/order and Git identities, and writes only a new sibling postrun directory. It imports no Torch and performs no tensor load, forward, fit or optimizer step.

Executed after formal completion (exit0):

```sh
scp -P 31192 /tmp/gate1a-v2-publication.hMhymv/postrun_v2_audit.py root@162.14.139.38:/root/gate1a_v2_postrun_audit.py
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 31192 root@162.14.139.38 '/root/.venvs/lcrseg-py310/bin/python /root/gate1a_v2_postrun_audit.py'
```

The successful report-copy command (no cloud writes) was:

```sh
set -o pipefail
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 31192 root@162.14.139.38 'tar -C /root/LCRSeg/runs/di_dmpa_gate1_v2/eaae37bbaa7546679d9e6893023afbeeef0ab5c6 --exclude=features --exclude=SHARED_GEOMETRY_SAMPLING_PLAN.json -czf - .' | tar -xzf - -C /Users/bominwang/Desktop/codes/SSL_CL_seg/experiments/lcrseg/docs/di_dmpa_jascl/gate1a_v2_results
```

The preceding unavailable-rsync and intentionally interrupted uncompressed-transfer events are disclosed in `GATE1A_V2_FAILURES_AND_WARNINGS.md`; neither was a formal diagnostic retry.

Raw attempt artifacts remain immutable. Public copies omit raw `.npy` feature arrays and the duplicate 44,276,416-byte sampling-plan JSON; their paths, sizes and hashes remain in the manifest. Tests, per-unit results, census, aggregate diagnostics, commands, logs and postrun verification are included.

The report commit is the commit first adding `GATE1A_V2_FINAL_REPORT.md` and this delivery manifest. Resolve its full SHA without a self-referential file hash:

```sh
git log --diff-filter=A --format=%H -- experiments/lcrseg/docs/di_dmpa_jascl/GATE1A_V2_FINAL_REPORT.md
```

Report publication uses the same v2 branch only. Main is neither merged nor updated. Final action: `STOP_FOR_INDEPENDENT_REVIEW`.

The pre-commit whitespace check preserves the standard CRLF bytes in raw CSVs:

```sh
git -c core.whitespace=cr-at-eol diff --cached --check
```

The staged blobs are separately SHA256-checked against every delivery-manifest entry; changing a raw CSV to suppress a formatting warning is not permitted.
