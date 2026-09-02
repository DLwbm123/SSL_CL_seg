# SHOR V0.4 fixed-policy held-out test report

## Outcome

**PASS_FIXED_POLICY_TEST_EFFECTIVENESS**

This was a new, one-shot held-out effectiveness question. It was not a V0.3.1 retry, repair, or re-adjudication. SHOR V0.3.1 remains permanently `FAIL_SELECTIVE_OVERRIDE_STABILITY`: H1/H2/H3/H4/H6 passed and H5 failed.

## Direct S3 versus S0 result

| Quantity | Point estimate | 95% paired hierarchical bootstrap CI |
|---|---:|---:|
| delta_all | 0.155844485481 | [0.141420521749, 0.168954953877] |
| delta_history | 0.234910501622 | [0.214103433180, 0.254091490096] |
| current_drop | 0.002287546802 | [0.000000000000, 0.006862640406] |

Positive seed deltas: 3/3. The fixed raw-value conditions were: `True`.

S3 overall foreground Dice was 0.753832513147; S0 was 0.597988027667. S3 historical-route frequency was 0.586666666667, false historical override frequency was 0.004444444444, and the S4 oracle gap was 0.014050089051.

## Isolation and scope

All test images were processed with the exact frozen descriptors, ridge state, temperature, thresholds, tie rule, score, S3 rule, snapshots, preprocessing, and deterministic inference settings. `TEST_CANDIDATE_SEAL.json` was durably written with zero test-GT reads, zero test-domain reads, and zero training steps before Phase B opened evaluator-only label/domain fields. S3 never received test-domain identity. No fitting, optimizer, backward, training, snapshot update, validation reuse, or H1-H6 rerun occurred.

The private NAS retains case-level metrics, image/label input hashes, descriptors, probabilities, routes, and masks. GitHub receives only aggregate metrics and small reviewer-facing evidence. The formal test is final; repeated evaluation is refused and no retry, redesign, training, or main-branch merge is authorized.
