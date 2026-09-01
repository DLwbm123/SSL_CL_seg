# SHOR-JASCL V0.3.1 Active-Support Fix final report

## Outcome

The one authorized zero-model-forward validation attempt completed with server-local child exit code 0 and scientific status `FAIL_SELECTIVE_OVERRIDE_STABILITY`. This is a scientific failure, not an engineering blocker. H1, H2, H3, H4, and H6 passed; H5 failed. Controls cannot rescue the primary S3 result.

The execution used exact published source commit `b7f85c7deaccbb8dbeef1ef998ac0823db55e75d`. The admission suite passed 194 tests, including 58 SHOR cases and all five registered active-support scenarios, with zero failures, errors, or skips.

## Active-support recovery

The scoped preflight ran only seed 0, stage 1, replicate 0. Of 279 train-memory rows, 179 were active and 100 had zero multiplicity. Every active OOF probability was finite; every inactive sentinel remained NaN; no inactive probability was recomputed. Threshold selection completed with a feasible threshold. Validation-data reads, validation-GT reads, model construction, and model forwards were zero.

The formal run then completed all 9 formal threshold units and all 45 bootstrap threshold units. The V0.3 numerical blocker did not recur. The repair changed only active-support compression in threshold/calibration entry points; descriptors, bootstrap draws, ridge and temperature grids, folds, SHOR scores, threshold rules, policies, and H1-H6 were unchanged.

## Registered gates

| Gate | Result | Registered evidence |
|---|---:|---|
| H1 calibration | pass | 9/9 formal units feasible and all finite |
| H2 current safety | pass | Current-domain drop `0.004436 <= 0.010`; maximum current-class drop `0.005414 <= 0.015`; maximum seed-domain drop `0.013308 <= 0.020` |
| H3 value | pass | Three-domain gain `0.138463`; historical gain `0.209912`; oracle gap `0.027164`; 3/3 positive seeds; positive REFUGE and RIM_ONE_r3 gains |
| H4 repair of soft failure | pass | Current and maximum-drop reductions `0.026495`; shared-gain loss `0.021577`; historical-gain loss `0.042216` |
| H5 stability | fail | Shared-gain p10 `0.098874` and historical-gain p10 `0.151029` passed; current-domain-drop p90 `0.029784 > 0.015`, maximum seed-domain-drop p90 `0.029784 > 0.025`, and two stage-2/domain-1 units had only 3/5 and 2/5 feasible replicates |
| H6 isolation | pass | Zero model construction/forward/autograd/backward/optimizers/training/test-GT reads; private input unchanged; validation GT evaluator-only after candidate seal |

## Coverage and archive

The run completed 1,116 closed-form ridge fits, 30 bootstraps, 915 formal routes, 3,660 formal candidate case-predictions, 4,575 bootstrap candidate case-predictions, 75 segmentation rows, 75 bootstrap metric rows, and 495 evaluator-only validation-GT case reads. No test evaluation occurred.

All nine stage barriers passed. The final create-only private bundle exactly covers 142 files and 1,215,417,402 bytes with content identity `34dc1cf9f27d0adc47a6166e9310dd227c4aaaac37bb658badfabc21c61018ac`. The original 183-file frozen input remained unchanged.

## Hard stop

Because the registered result is a scientific failure, the image-only domain-agnostic snapshot-routing line stops permanently under this protocol. No second attempt, threshold modification, validation refit, test evaluation, C0 regeneration, training, LoRA, adapter, Prostate, MnMS, sweep, or main merge was started or authorized.
