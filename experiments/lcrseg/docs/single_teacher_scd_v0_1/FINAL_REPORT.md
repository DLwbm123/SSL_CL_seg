# Single-Teacher SCD V0.1: INCOMPLETE_TRAINING_MATRIX

Shared stage0 and S/A/B/C/D stage1+2 completed. E stage1 failed at attempted update 1508 (epoch 48) with SCD root not bracketed; E stage2 did not start. The complete-matrix scientific gate was not run. This is an engineering terminal, not evidence for or against SCD's incremental value.

Completed stages contain 34,500 updates. Repaired E recorded 1,507 successful updates (latest durable checkpoint 1,504); the initial entropy-failed E attempt recorded 1. All formal attempts total 36,008, not the requested 39,800 complete matrix. The missing latest-trajectory budget is 3,793. Diagnostics replayed 1 + 3 updates separately; two exact-source qualification suites used 504 updates, and the synthetic resource probe used 1. Pre-freeze exploratory invocations are not claimed as fully measured compute.

The fixed solver exhausted 60 bracket doublings. Reproduction from the epoch47 checkpoint matched the three successful steps and failed at the next step. Accepting an unbracketed result because its residual is small would violate the explicit bracket requirement; no such fallback or threshold relaxation was applied.

## Available descriptive metrics

| Arm | F | H | N | BWT | Forget REFUGE | Forget RIM |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.585957181 | 0.525668818 | 0.704849150 | -0.247356877 | 0.215889510 | 0.278824244 |
| A | 0.573670633 | 0.503046191 | 0.709291739 | -0.270229288 | 0.197384175 | 0.343074402 |
| B | 0.503730021 | 0.417375901 | 0.692086118 | -0.357934587 | 0.281903295 | 0.433965878 |
| C | 0.684184175 | 0.747438542 | 0.630016095 | -0.025183331 | 0.042937733 | 0.007428929 |
| D | 0.718089435 | 0.742590645 | 0.698910789 | -0.043220135 | 0.029089197 | 0.057351073 |
| E | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

B minus S descriptive N difference: -0.012763031687; F difference: -0.082227160806. No preregistered gate is adjudicated on this incomplete matrix. E versus C/D and current-domain E adaptation are unavailable.

Current-domain rim/cup and full observed stage-domain trajectories are in PER_CLASS_METRICS.csv and STAGE_DOMAIN_MATRIX.csv. Absent E rows are not filled from its unfinished checkpoint.

S/B used one complete model; A used student plus current EMA; C/D/E used student plus immediate predecessor. All five completed arms passed independent student-only deployment with the teacher path hidden. E has no completed deployment artifact. MEMORY_AND_RUNTIME.csv contains measured peaks, tensor-state sizes and runtime; partial/failed attempts are separate rows.

No test, hidden unlabeled GT, replayed past training data, old formal_03, seed1/2 or external dataset was used. Historical locks and terminal results remain intact. These are researcher-exposed seed0 development results, not external confirmation.

Common and S/A/B/C/D source: 057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8. Repaired E source: 270e23985c8d78d0508fe6c4d43150f6ebffcc92. Private evidence: /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01. Images, labels, case identifiers, weights, private diagnostic vectors and full run logs are excluded from publication.

## Current-domain class metrics

| Arm | Stage1 rim | Stage1 cup | Stage2 rim | Stage2 cup |
|---|---:|---:|---:|---:|
| S | 0.732040818 | 0.674287966 | 0.667352519 | 0.745715298 |
| A | 0.734320250 | 0.673007673 | 0.668048777 | 0.761790257 |
| B | 0.745040953 | 0.670426999 | 0.606108818 | 0.746767704 |
| C | 0.720779549 | 0.683933944 | 0.538606357 | 0.576744528 |
| D | 0.754808275 | 0.702660849 | 0.703069330 | 0.635104701 |
| E | unavailable | unavailable | unavailable | unavailable |

## Observed training resources

Times are sums of logged completed epochs per arm, not parallel end-to-end wall time. Incremental arms exclude the once-trained common stage0. E is partial and its last three successful steps plus the failed step are outside its completed epoch timing/peak records.

| Arm | Full models | Peak allocated GiB | Peak reserved GiB | Logged epoch minutes |
|---|---:|---:|---:|---:|
| common | 1 | 0.477112 | 0.556641 | 48.683 |
| S | 1 | 0.477112 | 0.556641 | 18.284 |
| A | 2 | 0.885111 | 0.994141 | 33.872 |
| B | 1 | 0.875555 | 0.980469 | 28.478 |
| C | 2 | 0.924070 | 1.042969 | 46.155 |
| D | 2 | 0.924070 | 1.042969 | 28.881 |
| E | 2 | 0.924070 | 1.048828 | 14.897 |

The formal attempts account for 107,650 forwards, 36,010 supervised GAS autograd calls, 39 first-step branch diagnostic autograd calls, 36,008 backwards and 36,008 optimizer updates. Partial-attempt terminal counters are reconstructed from their durable checkpoint counters and the reproduced executed failure path; the complete-stage counters come directly from receipts. Qualification and diagnosis costs are separate. All five completed arms deploy one student; E has no final deployment.
