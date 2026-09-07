# CARe-HR V0.7.1 R2 — final capacity report

**Terminal: FAIL_FROZEN_ACTION_SPACE_CAPACITY. Engineering recovery and full evaluation completed successfully.**

O_CAP envelope achieves overall gain **0.01958758063839712** and historical gain **0.029381370957595684**, below the frozen **0.17 / 0.27** thresholds. These are Dice proportions: about 1.959 / 2.938 percentage points versus required 17 / 27 points. The original terminal function uses unrounded metrics, with no threshold tolerance. Relaxed-space results do not change this failure.

## Exact versions and unchanged science

- Requested predecessor:3505ffdfdeddb75d55e6fc982b78e20dafe6f816. Metadata-only publication commit bf88c3d365b981dba2f42f4fd8c2403ef6bb02f9 was also preserved.
- Branch: codex/care-hr-v0-7-1-capacity-audit.
- R1 action-generation source:9bbbacd25f3c3abf885205009eb14d698b36f32b.
- **R2 tested/evaluator source:271fc261380d2ef0abe5ea33088e8df72553eefe.** Later evidence/report commits are not the tested source.
- E0 qualification:6841c7dd23054f345b3c63fe652b74605f78c06b.
- E1 seal publication:d320dec355940e44047e96b326d14315bc8d475a.
- E2 preflight publication:bef853d539f672ba68215d522aab6485d41368c0.
- R1 parent seal SHA-256:9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4.
- R2 dual-source seal SHA-256:6bf0a8d6957a023590af224ea725e7f13757a1aa9a4a0c4587d2605f05a46aa0.

The adapter directly uses inherited di_dmpa_gate1.binding.safe_asset(DATA, relative), resolving DATA/h5/v1/relative with canonical containment. DATA, generic safe_path, execute_r1.py and all old science remain unchanged. The copied scientific orchestration is character-identical after removing the five explicitly audited I/O/accounting substitutions. Original scorer, action/proposal definitions, oracle selectors, tie-break, aggregation and terminal are reused without modification.

## Engineering qualification

Both local and server exact-source runs returned **189 passed, 0 failed, 0 errors, 0 skipped**: all 161 existing tests plus 28 new R2 tests. All 13 original A1 coverage items remain covered. An independent canonical-only fixture explicitly reproduces R1's FileNotFoundError and passes R2's actual loader. Wrong-root/decoy, multi-root, symlink safety, unreadable/read errors, hash/key/shape/dtype/value/HDF5 failures and legitimate all-ignore cases are covered.

Full 198-row synthetic production E1/E2/E3 tests cover completion, failure with actual parent exit 1, 198-row stat-error collection, precise 66-row domain materialization and read-only parent/seal protection. The expected old-error witness is not a failed or skipped R2 test. No loader or HDF5 decode is monkeypatched in these end-to-end runs. Test-only external fixture roots/identities do not create CLI population overrides.

## Real inputs and baseline parity

198 seed-case rows, 177 patients,66 rows per seed, nine probability caches and all parent identities verified. There are 177 physical GT files and 21 repeated seed-row uses. C6 retains 155 historical / 43 current routes. All 402 parent sealed files and original control/cache hashes verified unchanged. The 949 proposals and 15796 strict /26374 no-area /159232 free actions are reused in place, not regenerated.

E2 validated **198/198 canonical paths, zero exceptions**, with zero payload opens and zero domain materialization. Its admission binds the immutable dual-source seal and private resolved-asset allowlist. E3 then created a fresh reservation and process, reading original bytes, verifying SHA-256 and decoding the same bytes before unchanged shape/dtype/class validation.

**792 case-policy comparisons, 3168 scalar comparisons and 64 grouped control rows passed; maximum absolute difference=0.0**, with unchanged rtol0/atol1e-12. Main metrics average case scores inside seed/domain groups, then weight all nine groups equally; history uses six groups. Original fold/case/policy order is preserved. No pooled-confusion Dice or simple 198-row global mean replaces that call chain.

| Control | Overall macro Dice | Overall gain | Historical gain |
| --- | --- | --- | --- |
| current | 0.601673140 | 0.000000000 | 0.000000000 |
| frozen_Ridge_hard | 0.815629321 | 0.213956181 | 0.344010831 |
| frozen_SHOR | 0.805257917 | 0.203584777 | 0.313034848 |
| frozen_PPC_C6 | 0.802187013 | 0.200513873 | 0.308428491 |

## Complete oracle results

All 159232 sealed actions across 198 rows were scored. The four spaces, fixed lambdas and optimistic per-case envelope are separate. The table is rounded for readability; CSV files retain full precision and all class scores/drops, domains, seeds and differences versus PPC.

| Oracle | Overall macro Dice | Overall gain | Historical gain |
| --- | --- | --- | --- |
| O_CAP_lambda050 | 0.614967249 | 0.013294109 | 0.019941163 |
| O_CAP_lambda075 | 0.621107312 | 0.019434172 | 0.029151258 |
| O_CAP_envelope | 0.621260721 | 0.019587581 | 0.029381371 |
| O_SAFE0_lambda050 | 0.614940873 | 0.013267733 | 0.019901599 |
| O_SAFE0_lambda075 | 0.621080936 | 0.019407796 | 0.029111694 |
| O_SAFE0_envelope | 0.621234344 | 0.019561204 | 0.029341807 |
| O_NO_AREA_lambda050 | 0.760044873 | 0.158371733 | 0.237557600 |
| O_NO_AREA_lambda075 | 0.804766953 | 0.203093812 | 0.304640719 |
| O_NO_AREA_envelope | 0.807753361 | 0.206080221 | 0.309120331 |
| O_FREE_SUBSET_lambda050 | 0.762007196 | 0.160334056 | 0.240501084 |
| O_FREE_SUBSET_lambda075 | 0.807138381 | 0.205465241 | 0.308197861 |
| O_FREE_SUBSET_envelope | 0.810111398 | 0.208438258 | 0.312657387 |

O_CAP's overall/historical gain differences versus frozen PPC are -0.1809262921419793 /-0.2790471201092072. O_SAFE0 has almost the same capacity and still misses both value gates. Positive seed/history-domain gains do not satisfy the required gain magnitudes.

| Group | Macro gain | Rim gain | Cup gain |
| --- | --- | --- | --- |
| seed:0 | 0.034540021 | 0.020573052 | 0.048506991 |
| seed:1 | 0.005882606 | 0.010382799 | 0.001382413 |
| seed:2 | 0.018340115 | 0.013937357 | 0.022742873 |
| domain:Drishti_GS | 0.000000000 | 0.000000000 | 0.000000000 |
| domain:REFUGE | 0.041897197 | 0.040768670 | 0.043025724 |
| domain:RIM_ONE_r3 | 0.016865545 | 0.004124538 | 0.029606552 |

## Budget and safety attribution

Removing both area constraints while retaining count/class limits adds **0.18649264033298552** overall gain. Removing count/class limits as well adds only **0.0023580369360862575**. The two area constraints jointly account for most of the observed capacity gap; this experiment does not isolate either area's independent effect. Relaxed oracles remain diagnostics and do not replace the primary method.

| Envelope | Selected | No-op | Mask pixels | Hard changed | Probability changed | GT-valid hard changed |
| --- | --- | --- | --- | --- | --- | --- |
| O_CAP_envelope | 127 | 71 | 184697 | 159795 | 184697 | 159795 |
| O_SAFE0_envelope | 126 | 72 | 184474 | 159572 | 184474 | 159572 |
| O_NO_AREA_envelope | 154 | 44 | 1718901 | 1585441 | 1718901 | 1585441 |
| O_FREE_SUBSET_envelope | 154 | 44 | 1760183 | 1622888 | 1760183 | 1622888 |

There are 67 strict no-op-only spaces, 43 free no-op-only spaces and zero zero-current-foreground cases. Quantity/foreground-area/image-area rejection flags total 132858/72558/64706; they overlap and cannot be summed as disjoint rejections.

O_CAP selects 127 rows: all improve macro Dice, 126 have no class harm and one has a class tradeoff. O_SAFE0 selects 126, all satisfying its exact class nondecrease constraint. Each relaxed envelope selects 154, of which 151 have no class harm. Private Pareto output contains 207 nondominated action records. These retrospective GT-oracle fractions do not establish learned routing precision or calibrated safety.

## Coverage and sensitivity

The cohort contains 120 REFUGE, 48 RIM_ONE_r3 and 30 Drishti_GS seed-case rows. Historical coverage is 168 rows/149 patients; current-domain coverage 30 rows/28 patients. Every seed/domain group has valid support. There are **29196288 valid pixels,0 ignored pixels,0 all-ignore rows and0 all-ignore patients**. The frozen all-ignore convention was not invoked by real data.

Evaluable-only sensitivity exactly matches primary scores/gains (maximum difference 0); patient-equal means within original groups also match. The separately named global cross-seed patient-equal sensitivity first averages repeated seed observations per patient: O_CAP overall/historical gain is 0.029788715396721413 /0.035386594800132155 over 177/149 patients. Its different weighting does not replace or rescue the primary gate. All denominator/weighting tables are published separately.

All original full-draft gates are retained. Computable fields are descriptive oracle observations, not a full-method PASS. Bootstrap p90/p10, feasible refits and learned-consensus/calibration stability remain NOT_EVALUATED; no fitting was added to fill them.

## Access accounting, cost and final stop

R1's INCOMPLETE_EVALUATION, reservation and raw counters remain unchanged: 1 failed GT-open attempt, 0 successful reads/decodes, 66 domain rows materialized and 1 processed, 0 baseline/oracle rows. R2 records its new exposure separately; it does not reset the prior exposure.

R2 has **198 open attempts,198 successful opens,198 completed reads,198 hash passes,198 decode attempts/completions,198 validated seed-row uses,177 unique physical files decoded**. Repeated physical files were read per authorized seed-row and counted separately from unique files. Domain dataframe materialization/processing is 198/198. Baseline units are 792 case-policy /3168 scalar comparisons; oracle rows 198. The permitted frozen case-policy sidecar is separately used for parity and inherited C7 gap reporting.

Actual parent child exits for A1/E1/E2/E3 are all 0, and all children have terminated. Evaluator runtime 20.63651090802159 seconds; peak RSS 222712 KiB. New run artifacts total 4779295 bytes at measurement, excluding A1 scratch/source bundle and referenced parent caches.

**New sample-expert/batch forwards=0; model constructions/checkpoint tensor reloads=0; all real Ridge/PAV/temperature/router/risk-head/conformal fits=0; optimizer/segmentation/EMA/GAS/prototype updates=0.** R2 reuses 594 historical sample-expert outputs. Parameter/cache cost references are explicitly R1 historical observations. Evaluation is CPU-only.

All 150 protected files remain byte-identical locally and on the server, including the previous 99, R1 source/tests and all R1 reports. All 402 parent sealed files and R1 terminal/reservation reverified unchanged. The server checkout remains clean at 271fc261380d2ef0abe5ea33088e8df72553eefe. Old REVIEW_LOCK is unchanged; no main merge, formal_03 read or own-seed forbidden GT read occurred. Private IDs, per-case/Pareto contents, raw GT/images/probabilities/models and credentials are excluded from GitHub; private artifacts remain on NAS with published hashes.

This is the same development cohort with prior domain exposure and training/development use, not an independent external or unseen-patient generalization test.

**Stopped at the declared scientific terminal.** NEXT_STAGE_DRAFT.md contains only result-supported future questions. No subsequent fitting, training, action design or experiment was started.
