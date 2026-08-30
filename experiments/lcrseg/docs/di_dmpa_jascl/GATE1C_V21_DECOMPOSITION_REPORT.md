# Gate1C v2.1: fixed-pair gradient decomposition investigation

## Outcome

**NATIVE_DECOMPOSITION_FAILURE_REPRODUCED on both GPUs.** This is an
engineering investigation, not a Gate1C admission or a successful reproduction.
The original formal attempt remains **BLOCKED_INCOMPLETE_EVIDENCE**: 9/9
validation metric units (495 cases) and 25/72 draw0 pairs completed before the
unchanged gradient decomposition guard raised. No noise, posterior-mean or PoE
phase completed. Gate1B and overall Gate1 remain **FAIL_TRANSPORT_NOT_SUPPORTED**.

The sole failed pair, `B0/seed0/stage2/Drishti_GS/pair01`, was replayed exactly
once per physical GPU (0 and 1), with the original float32 engine, seeds,
sampling, objectives and tolerance. Both completed their forensic receipts at
2026-08-30T16:23:40 UTC. The complete numerical `details` objects, including
all forward/gradient hashes, are identical across devices.

## Numerical evidence

The first failing calculation is **R2 / class_balanced / global**. It is not
an R3 scientific-performance failure. All three violating coordinates are in
the encoder.

| Quantity | Both GPU replicas |
| --- | ---: |
| Active global gradient coordinates | 483,152 |
| Coordinates outside the original tolerance | 3 |
| Maximum absolute component-sum error | 3.1642557587474585e-06 |
| Maximum error / allowed componentwise bound | 1.4655137126931588 |
| Total gradient L2 norm | 8.692945358848743 |
| Relative L2 residual | 1.2563590963293842e-06 |
| Detached probability-leaf gradient sum error, float32 | 0.0 |
| Detached probability-leaf gradient sum error, float64 | 0.0 |

The worst tolerance-normalized coordinate is
`enc3.block.0.weight[28,27,1,2]`: native total `-0.010731285437941551`,
sum of class components `-0.010734324081568047`, allowed bound
`2.0734324081568046e-06`. All six individual block summaries and the global
summary are retained, including passing blocks; no coordinates were dropped.

Native autograd was **float32**. Converting its outputs to NumPy float64 does
not turn the preceding network differentiation into float64. Both TF32 switches
were false, AMP was off and strict deterministic algorithms were enabled.
The original predicate is still `allclose(total, component_sum, atol=1e-6,
rtol=1e-4)`, evaluated with the component sum as its reference.

The detached probability-leaf calculation is additive, while the discrepancy
appears after the network vector-Jacobian product. This supports a finite-
precision accumulation/cancellation explanation, but does **not yet establish
it with a high-precision network reference**. PyTorch's version-matched
[numerical-accuracy documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/docs/source/notes/numerical_accuracy.rst)
explains why mathematically equivalent float32 computation orders need not be
identical; that general fact alone is not proof of this particular cause.

## Provenance, checks and limitations

- Investigation preregistration: `6477a1c240a49c0c365217c16c6ff7ca0a5163e8`.
- Exact published inspector: `26121042817c0b7fc586ad44f869e86faca2c59b`.
- Shared diagnostic/training engine directories are byte-unchanged from the
  formal code `44a25254697fa535d2b48b64e27ecb226436f7d0`.
- Published-code synthetic suite: **121 passed, 0 failed, 0 skipped**, 12.682 s.
  `test_real.py` was explicitly excluded. The two real fixed-pair probes are
  separately reported here, not presented as a replacement for full integration.
- Both recorded student/teacher model guards passed bitwise before/after checks;
  all nine original B0 checkpoint hashes stayed unchanged. Native student
  `.grad` fields remained None. No optimizer, `.backward()`, EMA, GAS or
  prototype update was performed. No hidden/final-test GT was used.
- The original exception bypasses `probe_unit`'s later success-path bank/
  teacher-gradient isolation receipt. This inspector checks the model guard,
  checkpoint files and student gradient fields, but did **not** independently
  serialize a complete after-error in-memory legacy/current/history bank audit.
  Do not interpret these receipts as complete scientific isolation admission.
  A prospective reference probe must explicitly run that audit on captured
  exception-frame state, without modifying the old evidence.
- Initial development setup lacked the ignored `third_party` parent; it was
  created before tests/forwards. A macOS overlay archive emitted xattr warnings
  only in the disposable development checkout. The real probes used a separate
  clean detached checkout populated from the published Git bundle.

The original failed run, including all 648 independently rehashed files
(5,243,626,652 bytes), is preserved. Its manifest SHA256 is
`0d652551711e0a3ceff6ac8bdb0001355f4ec6083882460d740784ee837420d9`.
There was no new full Gate1C attempt, method registration, method optimizer
training, tolerance change, favorable-device selection or main-branch merge.

## Receipts and exact commands

Public metadata and JUnit files:
[gate1c_decomposition_results/6477a1c](gate1c_decomposition_results/6477a1c).
Raw tensors, images, labels and checkpoints are not published.

| File | SHA256 |
| --- | --- |
| GPU0 `INVESTIGATION_OUTCOME.json` | `6d47a9bb584b8ffbf265858ccd99ab823fce34d049cdebf0cef90609b1fb4e2d` |
| GPU1 `INVESTIGATION_OUTCOME.json` | `5607a2171510975948674ac043fc03bc807ac675c85e53afcd604a2a7a73d2ed` |
| Published-code JUnit | `a8210334305b268889101dd95f3b11dc87ef0aeccc76fe464012d9f6fc9e8d93` |
| Development-overlay JUnit | `1513faae0b51947b5834dee5a885220dc3ef055c77ca14dd24cb8e9d1eeed844` |

Remote checkout: `/root/SSL_CL_decomposition_2612104`.
Output prefix:
`/root/LCRSeg/runs/gate1c_decomposition_investigation/6477a1c240a49c0c365217c16c6ff7ca0a5163e8`.
Each `replica_gpu0` / `replica_gpu1` directory is create-only and occupied;
**do not rerun these commands into those directories**.

```sh
cd /root/SSL_CL_decomposition_2612104/experiments/lcrseg
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. \
/root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/di_dmpa_gate1c_v2 \
  --ignore=tests/di_dmpa_gate1c_v2/test_real.py \
  --junitxml=/root/gate1c_decomposition_tests_2612104.xml
```

The following was run concurrently once with `GPU=0` and once with `GPU=1`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH=/root/SSL_CL_decomposition_2612104/experiments/lcrseg \
/root/.venvs/lcrseg-py310/bin/python \
/root/SSL_CL_decomposition_2612104/experiments/lcrseg/scripts/inspect_gate1c_decomposition.py \
  --code-commit 26121042817c0b7fc586ad44f869e86faca2c59b --gpu "$GPU"
```

Next: publish a separate finite preregistration for a same-pair, same-random-
realization float64 **reference**, then its implementation/tests, before any
additional real forward. It must not relax the original guard, rescue the old
attempt, or launch a full diagnostic retry automatically.
