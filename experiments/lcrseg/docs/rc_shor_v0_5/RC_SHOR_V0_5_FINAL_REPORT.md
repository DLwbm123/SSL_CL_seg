# RC-SHOR V0.5 final report

## Outcome

**FAIL_RC_SHOR_VALUE**

RC-SHOR V0.5 used 49 seed-case observations from 37 leakage-free train-labeled patients in grouped five-fold outer OOF evaluation. SHOR V0.3.1 remains `FAIL_SELECTIVE_OVERRIDE_STABILITY` with H5 false; SHOR V0.4 remains `PASS_FIXED_POLICY_TEST_EFFECTIVENESS`.

## Gates

| Gate | Pass | Raw evidence |
|---|---:|---|
| isolation | True | V0.4 formal reads 0; all five seals preceded outer GT/domain; model updates 0 |
| value | False | overall gain 0.000000; historical 0.000000; REFUGE 0.000000; RIM 0.000000; seeds 0/3; oracle gap 0.246601 |
| current safety | True | current drop 0.000000; max current-class 0.000000; max seed-domain 0.000000 |
| stability | False | shared p10 0.000000; historical p10 0.000000; current p90 0.007658; max seed-domain p90 0.022973; feasible 5/100 |
| incremental | False | C6-C3 overall -0.217082; historical -0.325623; regret reduction -6.868819; p90 increase 0.007658 |

H5 recovery: **False**. Segmentation training/optimizer/update counts were 0; frozen cache materialization used 32 batches. Router fitting was closed-form with nonzero fit count and zero optimizer steps. C0-C8 exact aggregate values are in the status and metrics CSV. The five candidate-seal hashes, selected candidates, C3/C6 comparison, clustered bootstrap and ordinary-case sensitivity are in the status.

No V0.4 private test artifact was read, no old artifact was modified, no retry or main merge occurred, and no next stage was launched.
