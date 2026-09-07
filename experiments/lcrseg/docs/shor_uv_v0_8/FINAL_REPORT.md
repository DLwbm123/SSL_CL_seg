# SHOR-UV V0.8 development utility pilot

**Terminal: DEVELOPMENT_UTILITY_SIGNAL_NOT_ESTABLISHED**

This is a new development experiment. CARe-HR R2 remains FAIL_FROZEN_ACTION_SPACE_CAPACITY.
The deployed gate sees only 18 prediction/alpha features and the frozen SHOR route; it returns the frozen historical whole-image prediction or current. No oracle action or GT enters deployment.

Tested source: 8a4a9aa29364352f3a109f6f33e07db00a71b090. All five outer patient folds were deployed and sealed before this evaluation. Only the new heads, scaler, lambda and threshold exclude outer patients. The upstream segmenters, descriptor alpha and SHOR used these development patients previously. This is neither independent confirmation nor a strict historical-label-free online protocol.

| Policy | Overall Dice | Overall gain | Historical gain | History retention | Current drop | Current worst delta | Current delta < -0.10 | Fallback folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CURRENT | 0.601673140 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0 | 0 |
| RIDGE_HARD_FROZEN | 0.815629321 | 0.213956181 | 0.344010831 | 1.098953786 | 0.046153120 | -0.524484127 | 3 | 0 |
| SHOR_FROZEN | 0.805257917 | 0.203584777 | 0.313034848 | 1.000000000 | 0.015315364 | -0.459460914 | 1 | 0 |
| PPC_FROZEN | 0.802187013 | 0.200513873 | 0.308428491 | 0.985284843 | 0.015315364 | -0.459460914 | 1 | 0 |
| SHOR_CONFIDENCE_VETO | 0.790955060 | 0.189281920 | 0.291580562 | 0.931463587 | 0.015315364 | -0.459460914 | 1 | 0 |
| SHOR_UV_CF | 0.770180158 | 0.168507018 | 0.252760527 | 0.807451721 | 0.000000000 | 0.000000000 | 0 | 0 |
| SHOR_UV_ROUTED_ONLY | 0.782161766 | 0.180488626 | 0.278390621 | 0.889327889 | 0.015315364 | -0.459460914 | 1 | 0 |
| SHOR_UV_GAIN_ONLY | 0.652440931 | 0.050767791 | 0.083809369 | 0.267731754 | 0.015315364 | -0.459460914 | 1 | 4 |

## All primary numerical gates

| Category | Gate | Observed | Condition | Pass |
| --- | --- | ---: | --- | --- |
| VALUE | overall_gain | 0.16850701779 | >= 0.17 | False |
| VALUE | historical_gain | 0.252760526686 | >= 0.27 | False |
| VALUE | positive_REFUGE | 0.25484695726 | > 0.0 | True |
| VALUE | positive_RIM_ONE_r3 | 0.250674096111 | > 0.0 | True |
| VALUE | positive_seed_0 | 0.224492308994 | > 0.0 | True |
| VALUE | positive_seed_1 | 0.198104290328 | > 0.0 | True |
| VALUE | positive_seed_2 | 0.0829244540499 | > 0.0 | True |
| SAFETY | current_mean_drop | 0 | <= 0.01 | True |
| SAFETY | current_maximum_class_drop | 0 | <= 0.015 | True |
| SAFETY | maximum_seed_domain_mean_drop | 0 | <= 0.02 | True |
| SAFETY | current_case_delta_below_minus_010 | 0 | == 0 | True |
| BASELINE | overall_difference_from_SHOR | -0.0350777595382 | >= -0.01 | False |
| BASELINE | historical_difference_from_SHOR | -0.0602743212035 | >= -0.02 | False |
| INCREMENTAL | current_drop_reduction_from_SHOR | 0.0153153637925 | >= 0.005 | True |

Simple-veto comparison: **MIXED_TRADEOFF_OR_POSSIBLE_INCREMENTAL_VALUE**. The flag requires confidence veto to weakly dominate primary overall/history gains and all four safety coordinates; otherwise the tradeoff is reported without claiming superiority.

## Fit and evidence boundaries

FIT_ACCOUNTING.json reports every actual weighted Ridge design solve and three scalar outputs. CF and GAIN_ONLY share identical heads; ROUTED_ONLY is fitted separately. Each participating patient has total weight 1 across seed/expert duplicates. Standardization is weighted and fit-split-only. Lambda MSE uses clipped inner OOF predictions; exact ties select larger lambda.
INNER_SELECTION_SUMMARY.json exposes every fold choice/fallback. Full private inner candidate tables and models are sealed on NAS. A fallback emits current and remains in all denominators.
BASELINE_PARITY.json checks 792 case-policy / 3168 scalar comparisons and inherited grouped controls at rtol=0, atol=1e-12. New deployed masks are checked byte-for-byte against the selected cached expert argmax.
The target service read only the 198 authorized own-seed train_labeled labels using the frozen R2 original-bytes/hash/decode loader. Fit services received only their own outer-training package, and fit calls only inner-training pairs. Deployment read no GT/domain fields. Outer evaluation reused target-service metrics after verifying all prediction seals.
SAFETY_AND_TAIL.csv separates seed rows from unique patients, source mismatches from segmentation harm, and macro-beneficial from class-nonharm acceptance. HEAD_PREDICTABILITY.json reports patient-weighted MAE, nonzero gain-sign accuracy and harmful-pair recall. Zero targets are separate.
PATIENT_BOOTSTRAP.csv contains 2000 fixed-policy paired patient-cluster draws with seed 2026090801, including invalid empty-group counts and no redraws. These intervals do not measure retraining stability; old router-refit p90/p10 remain NOT_EVALUATED.
Current-domain support is only 30 seed rows / 28 patients. Zero observed extreme losses cannot establish a strong tail-safety guarantee.
No new image reads, segmentation forward, checkpoint tensor loading, optimizer/backward, EMA/GAS/prototype update, conformal fitting, old formal_03 or forbidden GT access occurred. Old protected source and R1/R2 artifacts were verified unchanged.

## Stop

All fixed controls and ablations are complete. No outer-result-driven recovery, tuning, new feature, threshold change, new experiment or external test was started. Any further work requires a separately frozen protocol and independent confirmation.


## Interpretation of the completed fixed matrix

The primary gate vetoed 25 of SHOR's 158 historical routes and accepted 133. All 133 accepted rows improved macro Dice, but six had a rim/cup class decrease; observed macro-benefit precision 133/133 must not be described as perfect class safety. Of the 25 vetoes, 24 discarded macro-beneficial routes. These are actual sealed-model decisions, not capacity-oracle precision.

CF removes the single current-domain historical override, changing worst current seed-row macro delta from -0.45946091377429465 to 0. Its current mean/class drops are 0 and no current row has delta < -0.10. It retains only 80.74517210785936% of frozen SHOR historical gain, so both VALUE targets and both BASELINE noninferiority targets fail. The point result is not rounded up to 0.17.

Confidence veto retains 93.14635872778673% of history gain and achieves overall/history gains 0.18928192030774318 / 0.291580562357853, but preserves the same current-domain large harm. ROUTED_ONLY likewise retains that harm despite stronger value than CF. Neither is an admissible replacement winner for the prespecified primary. GAIN_ONLY falls back to current in four folds; the remaining fold retains the large current harm. CF itself has no fallback fold.

Actual private identity checks confirm that frozen SHOR and PPC's current-domain error is the same one seed-case and one patient. The complete current/history harm tables were inspected, not inferred from a routing-accuracy ratio. For this error, CF still predicts a positive macro gain; its H threshold causes the veto. ROUTED_ONLY underestimates that harm enough to accept, and GAIN_ONLY has no harm veto. Individual identities and head predictions remain private. This is one retrospective mechanism witness, not a general causal or tail-safety claim.

Across all 396 pairs there are 137 class-harm pairs. Current-domain pairs contribute 59/60 harmful pairs, whereas frozen SHOR selected only one of these 60 pairs for its original historical route. CF outer patient-weighted MAE is 0.18923731329185148 for rim gain, 0.17563811597347045 for cup gain, and 0.09985987772992758 for H; nonzero gain-sign accuracy is 0.8573446327683616. Better target prediction than ROUTED_ONLY does not establish adequate deployed utility.

All 2000 patient-bootstrap draws had valid groups. CF overall gain 95% interval is [0.14787801985945645, 0.19012440239620504], with paired SHOR difference [-0.055899436986519134, -0.01448244908529567]. Historical gain interval is [0.22181702978918463, 0.28518660359430753], with paired SHOR difference [-0.08988558899411271, -0.03466648881655677]. They quantify fixed-policy cohort resampling only.

Local and server exact-source tests each passed 210/210 (189 inherited plus 21 new), zero failures/errors/skips. Fourteen study child processes and the server test child all exited 0 and have terminated. Raw fit event logs independently record 170 attempts, 170 completed three-output Ridge designs (510 scalar heads), zero failed fits. The complete study took 63.39316987991333 seconds, producing 128431482 bytes in run_01 at closeout; test scratch/source bundle are excluded. New network forwards/training/image reads remain zero.
