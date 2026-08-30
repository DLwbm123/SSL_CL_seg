# Gate1C v2.1: same-draw float64 reference report

## Outcome and limits

**SAME_PAIR_FP64_NUMERICAL_REFERENCE_SUPPORTED on both GPUs.** Both native
float32 failures reproduced with exactly the previously recorded tensor and
gradient hashes. The isolated float64 calculation used the same captured
float32 Gaussian realization, original teacher target, R2 weights and class
strata. All seven blocks passed the **unchanged** decomposition tolerance.

This supports finite-precision network VJP accumulation as the explanation
for this pair's discrepancy, rather than a nonadditive objective partition.
It does not establish behavior of the remaining 71 pairs, approve full-run
precision changes, or demonstrate that DI-DMPA reproduces successfully.
Original Gate1C v2.1 remains **BLOCKED_INCOMPLETE_EVIDENCE**; Gate1B and overall
Gate1 remain **FAIL_TRANSPORT_NOT_SUPPORTED**. No scientific gate was rescued.

## Results

The entire reference-detail objects and three native draw hashes are identical
across GPUs, not just their displayed rounded statistics.

| Quantity | Native float32 | Same-draw float64 reference |
| --- | ---: | ---: |
| Active gradient coordinates | 483,152 | 483,152 |
| Coordinates outside original tolerance | 3 | 0 |
| Maximum component-sum absolute error | 3.1642557587474585e-06 | 4.440892098500626e-15 |
| Relative L2 component-sum residual | 1.2563590963293842e-06 | 1.7027095235575764e-15 |
| Total gradient L2 norm | 8.692945358848743 | 8.692941260159767 |

Native versus reference total-gradient relative L2 difference is
**1.003654873528068e-06** (registered maximum 1e-3), cosine
**0.9999999999996072** (registered minimum 0.9999).
The float64 decomposition residual is below the registered reference bound
1e-10. All six parameter blocks and all three class-component comparisons
are retained in the receipts; only the global summary is displayed here.

Forward values are close, but **not bitwise equal**: maximum absolute logit
difference 0.00012327763365860278; feature difference
2.949584288813867e-06; probability difference 8.572690944785855e-06.
Their relative L2 differences are respectively 2.9514880874847427e-07,
2.4332915746304857e-07 and 3.64592473960398e-07. The reference is a different
precision calculation, not an assertion of identical native arithmetic.

## Provenance and isolation

- Preregistration: `136f19fd9b4ba75dc8f4891e4d7601c58d7d90fb`, published
  before reference implementation. MD SHA256:
  `d6950a517c540a83d0018972f67ee9462bb7d619d58b59a042dcc3181a691104`;
  JSON SHA256:
  `b6933c33ae5425d73818db9c22859bdcd4f3b84c5ac5363412dd36a2e89c0824`.
- Exact published helper/tests: `d87b7cb7af2802a7d09cab7d4231794a0de69815`.
  Shared engines remain byte-unchanged from formal code `44a2525`.
  Official tracked classifier source remains commit
  `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`; its source file SHA256 is
  `4f91d0e3465ef0e34218e8a5138e0643acd46f14a1e87adf20829cf163428343`.
- Clean published-code synthetic tests: **126 passed, 0 failed, 0 skipped**,
  13.991 s. `test_real.py` was explicitly excluded; the two real reference
  probes are reported separately and are not full Gate1C integration coverage.
- GPU0: 2026-08-30T16:43:46.571811 to 16:43:51.810020 UTC (5.238209 s).
  GPU1: 16:43:46.841703 to 16:43:51.964085 UTC (5.122382 s). Both exited 0.
  Each performed 3 native forwards + 1 reference forward and four reference
  VJPs: **8 real model forwards total**, exactly one registered pair per GPU.
- Original failure guard raised on both devices. No guard was patched,
  residual corrected, total defined from component sums, coordinate dropped,
  class stratum changed or favorable device selected.
- Original student/teacher model guards passed. This new probe explicitly ran
  the existing isolation helper on captured exception state before and after
  its reference; legacy/current/history hashes matched. Original and shadow
  model/classifier/GAS/buffer states were unchanged within their respective
  dtypes. Teachers were frozen, inactive gradients were None, and every
  parameter.grad field remained None. Reference execution did not advance
  Python, NumPy, CPU or CUDA RNG states.
- All nine original checkpoint hashes and original formal failure/status/
  manifest hashes stayed unchanged. No optimizer, backward, EMA, GAS or
  prototype update occurred; no hidden/final-test GT was used. Prior baseline
  bank recovery's 400 supervised updates remain separately disclosed.

The current probe's complete after-error isolation evidence does not fill
the omitted receipt in the earlier native-only investigation. Old evidence is
preserved as originally recorded. No full retry, new method training, method
registration or main merge occurred.

## Artifacts and commands

Public metadata/JUnit:
[gate1c_fp64_reference_results/136f19f](gate1c_fp64_reference_results/136f19f).
Only metadata, aggregate differences, hashes and tests are published; no raw
images, labels, feature/gradient arrays or checkpoints.

| File | SHA256 |
| --- | --- |
| GPU0 `REFERENCE_OUTCOME.json` | `03b09db1996adefe098210edff4f01b6968b80258ef0737e08267f832621660c` |
| GPU1 `REFERENCE_OUTCOME.json` | `761e977f7ddc102e8e84cfc5735813af3f8a6ca94b8a22b029237f1b555e81d7` |
| Exact-code JUnit | `359db4eac99c153676c655572a9d48393d6fdd966bca27913ec3207bcd83f1d8` |
| Development JUnit | `bdbf28ca7e6c9e10a8afd0aeeda5819e64b0c7d10a9640eacc7e7acd9957344d` |

Remote code: `/root/SSL_CL_fp64_d87b7cb`, clean detached code commit.
Output prefix:
`/root/LCRSeg/runs/gate1c_fp64_reference/136f19fd9b4ba75dc8f4891e4d7601c58d7d90fb`.
Both replica directories are occupied and create-only; **do not repeat** these
commands. Published code was transferred with verified Git bundle SHA256
`fc0c4b1e06897c06a2218a019166cf2df84c483e6fc6a65a5c05c9ec35b407ae`.
The development-only tar overlay emitted macOS xattr warnings; the exact-code
test and real runs used the clean Git-bundle checkout, not that overlay.

```sh
cd /root/SSL_CL_fp64_d87b7cb/experiments/lcrseg
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. \
/root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/di_dmpa_gate1c_v2 \
  --ignore=tests/di_dmpa_gate1c_v2/test_real.py \
  --junitxml=/root/gate1c_fp64_tests_d87b7cb.xml
```

The following ran concurrently once with `GPU=0` and once with `GPU=1`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH=/root/SSL_CL_fp64_d87b7cb/experiments/lcrseg \
/root/.venvs/lcrseg-py310/bin/python \
/root/SSL_CL_fp64_d87b7cb/experiments/lcrseg/scripts/reference_gate1c_fp64.py \
  --code-commit d87b7cb7af2802a7d09cab7d4231794a0de69815 --gpu "$GPU"
```

Latest post-run resource check: both GPUs idle, no reference worker remaining;
available persistent disk 9,767,464,960 bytes. This is not a claim of sustained
maximum GPU utilization or sufficient space for another full diagnostic.

## Next finite iteration

Prepare an independent **diagnostic-precision** amendment, not a training
change. Reuse the existing engine. Keep native float32 teacher probabilities,
features, PAS/reliability weights, masks, class strata and stochastic draws;
investigate an isolated same-draw float64 student gradient receiver for both
labeled and unlabeled VJPs. Preserve the original allclose tolerance and all
scientific C conditions, and explicitly identify the revised numeric version.

Before any further real forward, publish the amended protocol, exact code and
tests, including the draw-replay contract for both student forwards and all
teacher/PoE targets. Start with a finite integration pilot, not a full run.
Require native output/weight parity, shadow/source immutability, gradient
partition/decomposition and strict determinism before scaling. Do not change
the training optimizer, EMA behavior, prototype banks or scientific thresholds.

Separately inspect disk headroom and peak artifact size before authorizing a
full diagnostic; do not delete old evidence or assume the current free space
is enough. Any cache-reuse/storage change must be explicit and prospective.
The long-running goal and same-thread follow-up remain active for the current
method only; this finite numerical-reference milestone is complete, but the
method reproduction is not.
