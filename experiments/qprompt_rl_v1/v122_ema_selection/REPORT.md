# V122 independent EMA conflict selection — complete

Primary status: **DEVELOPMENT_EMA_CONFLICT_SIGNAL**. Independent EMA_ALL rule quality gate: **True**. EMA_HALF quality gate: **True**; EMA ranking over matched random: **False**; prototype ranking increment: **False**. These separate frozen outcomes are all reported; none establishes model-held-out generalization, actual training/Dice gain or RL/campaign success.

Completed 2026-10-09T17:57:59.651309+08:00, elapsed 53.8seconds, rootEXIT0 and once-only completion auditPASS. Run `v122_ema_selection_20261009T095622Z`; execution/preregistration `5cdd7507b4473f798d45354a9af653a0f44bfb2f` proxy-pushed with remote/anonymous200 verified before launch. GPU4/NASprobe/neutralargv/EXEC_RUN no-overlapPASS. All160forward pairs,40support exclusions,40metric rows,8control summaries and readout recomputation audited; all models/EMA/memory/optimizer states unchanged. BASE rows exactly reproduce V121, an internal consistency check, not independent replication.

## Independent candidate and quota

CE is the original selected set restricted to observable mixed-target→EMA class conflicts1→0 or2→1. These two directions came from V120 development evidence; no new population validation is claimed. BASE retains the original target. EMA_ALL corrects every CE location. Within each image and each direction, K=floor(CEcount/2) is independently determined from EMA and mixed target, never from prototype counts. EMA_HALF ranks by EMA winning-minus-runnerup probability; RANDOM_HALF uses one fixed generator permutation; PROTO_HALF ranks by cosine similarity to the EMA class prototype minus best other class prototype. All use the same probability exchange toward EMA, not the full EMA soft vector. Stable flat-index ties, no threshold or minimum-one rule, no alternative budget/seed search.

The standalone EMA/random mask function has no prototype/label input; its output remains identical under changed synthetic prototype scores. Three HALF arms match actual count, destination and original EMA class-weight sums exactly. EMA_ALL has intentionally different coverage and is not a matched-budget ranking arm. Continuous confidence/target-amplitude is not matched; weightedL1is reported. Prototype ranking may select negative-margin positions to meet K rather than silently changing the budget.

## Every arm versus BASE

Rates are equal-context effects over the full original selected set. Positive weighted target accuracy, negative true-label NLL delta are better; values are fractions/log units, not Dice. Counts pool correlated pixel exposures and weight large-image-count conditions more, so they do not replace the primary means.

| arm | changed | repaired | harmed | both_wrong_changed | weighted_accuracy_delta | weighted_nll_delta | joint_positive_contexts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BASE | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0 |
| EMA_ALL | 15791 | 14272 | 1519 | 0 | 0.005896418 | -0.005664188 | 8 |
| EMA_HALF | 7874 | 7290 | 584 | 0 | 0.003185625 | -0.002695843 | 8 |
| RANDOM_HALF | 7874 | 7118 | 756 | 0 | 0.002934233 | -0.002809685 | 8 |
| PROTO_HALF | 7874 | 7329 | 545 | 0 | 0.003204395 | -0.002753055 | 8 |

## All six frozen comparisons

Each comparison requires both mean directions, jointly positive in≥6/8contexts, and both directions in every n=2/n=8/source2000/source8000 subgroup. Ties and zero effects fail strict positivity and remain in the denominator. Primary EMA_ALL−BASE cannot be replaced by a favorable half-budget result. Prototype increment requires all its three comparisons(BASE,EMA_HALF,RANDOM_HALF); it need not invent a different destination class. No historical practical/absolute/matched-CE/fresh-stream gate changes.

| method | control | equal_context_weighted_accuracy_delta | equal_context_weighted_nll_delta | joint_positive_contexts | passed |
| --- | --- | --- | --- | --- | --- |
| EMA_ALL | BASE | 0.005896418 | -0.005664188 | 8 | True |
| EMA_HALF | BASE | 0.003185625 | -0.002695843 | 8 | True |
| EMA_HALF | RANDOM_HALF | 0.000251392 | 0.000113842 | 2 | False |
| PROTO_HALF | BASE | 0.003204395 | -0.002753055 | 8 | True |
| PROTO_HALF | EMA_HALF | 0.000018770 | -0.000057212 | 3 | False |
| PROTO_HALF | RANDOM_HALF | 0.000270162 | 0.000056630 | 3 | False |

### EMA_ALL minus BASE

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.005276919 | -0.003828416 |
| 1 | 2000 | 2 | 0.005924477 | -0.005863444 |
| 2 | 2000 | 8 | 0.002571272 | -0.002436693 |
| 3 | 2000 | 8 | 0.003033730 | -0.002582616 |
| 4 | 8000 | 2 | 0.010241788 | -0.010195314 |
| 5 | 8000 | 2 | 0.010560820 | -0.010732529 |
| 6 | 8000 | 8 | 0.005376460 | -0.005472236 |
| 7 | 8000 | 8 | 0.004185876 | -0.004202257 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.008001001 | -0.007654926 |
| labeled_images=8 | 0.003791834 | -0.003673450 |
| source_step=2000 | 0.004201599 | -0.003677792 |
| source_step=8000 | 0.007591236 | -0.007650584 |

### EMA_HALF minus BASE

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.002821428 | -0.001575310 |
| 1 | 2000 | 2 | 0.003070581 | -0.002682409 |
| 2 | 2000 | 8 | 0.001291083 | -0.001124221 |
| 3 | 2000 | 8 | 0.001466394 | -0.001051821 |
| 4 | 8000 | 2 | 0.005466830 | -0.004567297 |
| 5 | 8000 | 2 | 0.006176288 | -0.005791527 |
| 6 | 8000 | 8 | 0.002854900 | -0.002688577 |
| 7 | 8000 | 8 | 0.002337497 | -0.002085582 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.004383782 | -0.003654135 |
| labeled_images=8 | 0.001987468 | -0.001737550 |
| source_step=2000 | 0.002162372 | -0.001608440 |
| source_step=8000 | 0.004208878 | -0.003783246 |

### EMA_HALF minus RANDOM_HALF

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.000233351 | 0.000213879 |
| 1 | 2000 | 2 | 0.000179690 | 0.000119916 |
| 2 | 2000 | 8 | -0.000019067 | 0.000110990 |
| 3 | 2000 | 8 | -0.000035466 | 0.000234501 |
| 4 | 8000 | 2 | 0.000356418 | 0.000555331 |
| 5 | 8000 | 2 | 0.000843179 | -0.000344277 |
| 6 | 8000 | 8 | 0.000182800 | 0.000034434 |
| 7 | 8000 | 8 | 0.000270231 | -0.000014040 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000403160 | 0.000136212 |
| labeled_images=8 | 0.000099624 | 0.000091471 |
| source_step=2000 | 0.000089627 | 0.000169821 |
| source_step=8000 | 0.000413157 | 0.000057862 |

### PROTO_HALF minus BASE

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.002948710 | -0.001822852 |
| 1 | 2000 | 2 | 0.003070581 | -0.002779718 |
| 2 | 2000 | 8 | 0.001413655 | -0.001211503 |
| 3 | 2000 | 8 | 0.001466394 | -0.001020670 |
| 4 | 8000 | 2 | 0.005403932 | -0.004523668 |
| 5 | 8000 | 2 | 0.006134129 | -0.005836367 |
| 6 | 8000 | 8 | 0.002862965 | -0.002703084 |
| 7 | 8000 | 8 | 0.002334794 | -0.002126579 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.004389338 | -0.003740651 |
| labeled_images=8 | 0.002019452 | -0.001765459 |
| source_step=2000 | 0.002224835 | -0.001708686 |
| source_step=8000 | 0.004183955 | -0.003797424 |

### PROTO_HALF minus EMA_HALF

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.000127282 | -0.000247542 |
| 1 | 2000 | 2 | 0.000000000 | -0.000097309 |
| 2 | 2000 | 8 | 0.000122571 | -0.000087281 |
| 3 | 2000 | 8 | 0.000000000 | 0.000031151 |
| 4 | 8000 | 2 | -0.000062897 | 0.000043629 |
| 5 | 8000 | 2 | -0.000042159 | -0.000044840 |
| 6 | 8000 | 8 | 0.000008065 | -0.000014507 |
| 7 | 8000 | 8 | -0.000002702 | -0.000040997 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000005557 | -0.000086516 |
| labeled_images=8 | 0.000031983 | -0.000027909 |
| source_step=2000 | 0.000062463 | -0.000100245 |
| source_step=8000 | -0.000024923 | -0.000014179 |

### PROTO_HALF minus RANDOM_HALF

| context | source_step | labeled_images | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0.000360634 | -0.000033663 |
| 1 | 2000 | 2 | 0.000179690 | 0.000022606 |
| 2 | 2000 | 8 | 0.000103505 | 0.000023708 |
| 3 | 2000 | 8 | -0.000035466 | 0.000265653 |
| 4 | 8000 | 2 | 0.000293521 | 0.000598960 |
| 5 | 8000 | 2 | 0.000801020 | -0.000389118 |
| 6 | 8000 | 8 | 0.000190864 | 0.000019927 |
| 7 | 8000 | 8 | 0.000267529 | -0.000055036 |

| group | weighted_accuracy_delta | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000408716 | 0.000049697 |
| labeled_images=8 | 0.000131608 | 0.000063563 |
| source_step=2000 | 0.000152090 | 0.000069576 |
| source_step=8000 | 0.000388234 | 0.000043683 |

## Full candidate counts, quotas and overlaps

Each quota was computed per image before aggregation. Floor(sum/2) can differ from the sum of image-level floors; no posthoc quota adjustment was made. Prototype inputs do not determine CE or quota. Unlike V121's shuffle, RANDOM_HALF does not inherit per-image intervention counts from prototypes. All CE pixels are enumerated, including strata in which the old prototype candidate had no positives.

| context | selected | conflict_1_0 | conflict_2_1 | conflict_total | quota_1_0 | quota_2_1 | half_budget | EMA_PROTO_overlap | RANDOM_PROTO_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 146094 | 201 | 505 | 706 | 100 | 252 | 352 | 290 | 177 |
| 1 | 146160 | 51 | 559 | 610 | 25 | 279 | 304 | 284 | 155 |
| 2 | 583423 | 2860 | 106 | 2966 | 1428 | 51 | 1479 | 1348 | 765 |
| 3 | 583570 | 1314 | 638 | 1952 | 655 | 316 | 971 | 823 | 481 |
| 4 | 146979 | 724 | 739 | 1463 | 361 | 369 | 730 | 667 | 360 |
| 5 | 146984 | 324 | 1011 | 1335 | 162 | 505 | 667 | 596 | 324 |
| 6 | 587956 | 3306 | 963 | 4269 | 1650 | 479 | 2129 | 1997 | 1059 |
| 7 | 588011 | 926 | 1564 | 2490 | 462 | 780 | 1242 | 1106 | 616 |

| context | BASE_changed | EMA_ALL_changed | EMA_HALF_changed | RANDOM_HALF_changed | PROTO_HALF_changed |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 706 | 352 | 352 | 352 |
| 1 | 0 | 610 | 304 | 304 | 304 |
| 2 | 0 | 2966 | 1479 | 1479 | 1479 |
| 3 | 0 | 1952 | 971 | 971 | 971 |
| 4 | 0 | 1463 | 730 | 730 | 730 |
| 5 | 0 | 1335 | 667 | 667 | 667 |
| 6 | 0 | 4269 | 2129 | 2129 | 2129 |
| 7 | 0 | 2490 | 1242 | 1242 | 1242 |

| context | BASE_changed_weight | EMA_ALL_changed_weight | EMA_HALF_changed_weight | RANDOM_HALF_changed_weight | PROTO_HALF_changed_weight |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.000000000 | 605.500000000 | 302.000000000 | 302.000000000 | 302.000000000 |
| 1 | 0.000000000 | 584.500000000 | 291.500000000 | 291.500000000 | 291.500000000 |
| 2 | 0.000000000 | 1536.000000000 | 765.000000000 | 765.000000000 | 765.000000000 |
| 3 | 0.000000000 | 1295.000000000 | 643.500000000 | 643.500000000 | 643.500000000 |
| 4 | 0.000000000 | 1101.000000000 | 549.500000000 | 549.500000000 | 549.500000000 |
| 5 | 0.000000000 | 1173.000000000 | 586.000000000 | 586.000000000 | 586.000000000 |
| 6 | 0.000000000 | 2616.000000000 | 1304.000000000 | 1304.000000000 | 1304.000000000 |
| 7 | 0.000000000 | 2027.000000000 | 1011.000000000 | 1011.000000000 | 1011.000000000 |

| context | BASE_weighted_target_L1_change | EMA_ALL_weighted_target_L1_change | EMA_HALF_weighted_target_L1_change | RANDOM_HALF_weighted_target_L1_change | PROTO_HALF_weighted_target_L1_change |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.000000000 | 416.353822559 | 161.609894156 | 201.069976717 | 173.176495314 |
| 1 | 0.000000000 | 521.263467401 | 235.551664650 | 254.658754468 | 243.235513806 |
| 2 | 0.000000000 | 1284.502982557 | 551.884449899 | 637.005638510 | 554.064853996 |
| 3 | 0.000000000 | 1005.977478325 | 434.830117375 | 498.729102343 | 423.398116320 |
| 4 | 0.000000000 | 988.721134633 | 418.972832471 | 496.668202251 | 418.269196540 |
| 5 | 0.000000000 | 1096.966987073 | 506.493534863 | 552.197846413 | 511.591900796 |
| 6 | 0.000000000 | 2404.948014766 | 1106.496100485 | 1202.836170882 | 1108.406579435 |
| 7 | 0.000000000 | 1865.685924262 | 840.611926436 | 926.334030062 | 848.851497829 |

## All per-context metrics

Full additive unweighted/weighted rates and counts are in RESULTS.json/csv; all40arm/context rows appear below.

| context | source_step | labeled_images | scope | changed | repaired | harmed | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 0 | 2000 | 2 | EMA_ALL | 706 | 652 | 54 | 0.005276919 | -0.003828416 |
| 0 | 2000 | 2 | EMA_HALF | 352 | 334 | 18 | 0.002821428 | -0.001575310 |
| 0 | 2000 | 2 | RANDOM_HALF | 352 | 323 | 29 | 0.002588077 | -0.001789189 |
| 0 | 2000 | 2 | PROTO_HALF | 352 | 340 | 12 | 0.002948710 | -0.001822852 |
| 1 | 2000 | 2 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | EMA_ALL | 610 | 595 | 15 | 0.005924477 | -0.005863444 |
| 1 | 2000 | 2 | EMA_HALF | 304 | 303 | 1 | 0.003070581 | -0.002682409 |
| 1 | 2000 | 2 | RANDOM_HALF | 304 | 293 | 11 | 0.002890891 | -0.002802324 |
| 1 | 2000 | 2 | PROTO_HALF | 304 | 303 | 1 | 0.003070581 | -0.002779718 |
| 2 | 2000 | 8 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | EMA_ALL | 2966 | 2377 | 589 | 0.002571272 | -0.002436693 |
| 2 | 2000 | 8 | EMA_HALF | 1479 | 1188 | 291 | 0.001291083 | -0.001124221 |
| 2 | 2000 | 8 | RANDOM_HALF | 1479 | 1197 | 282 | 0.001310150 | -0.001235211 |
| 2 | 2000 | 8 | PROTO_HALF | 1479 | 1234 | 245 | 0.001413655 | -0.001211503 |
| 3 | 2000 | 8 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 3 | 2000 | 8 | EMA_ALL | 1952 | 1857 | 95 | 0.003033730 | -0.002582616 |
| 3 | 2000 | 8 | EMA_HALF | 971 | 918 | 53 | 0.001466394 | -0.001051821 |
| 3 | 2000 | 8 | RANDOM_HALF | 971 | 923 | 48 | 0.001501860 | -0.001286323 |
| 3 | 2000 | 8 | PROTO_HALF | 971 | 917 | 54 | 0.001466394 | -0.001020670 |
| 4 | 8000 | 2 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 4 | 8000 | 2 | EMA_ALL | 1463 | 1400 | 63 | 0.010241788 | -0.010195314 |
| 4 | 8000 | 2 | EMA_HALF | 730 | 716 | 14 | 0.005466830 | -0.004567297 |
| 4 | 8000 | 2 | RANDOM_HALF | 730 | 699 | 31 | 0.005110411 | -0.005122628 |
| 4 | 8000 | 2 | PROTO_HALF | 730 | 713 | 17 | 0.005403932 | -0.004523668 |
| 5 | 8000 | 2 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 5 | 8000 | 2 | EMA_ALL | 1335 | 1243 | 92 | 0.010560820 | -0.010732529 |
| 5 | 8000 | 2 | EMA_HALF | 667 | 667 | 0 | 0.006176288 | -0.005791527 |
| 5 | 8000 | 2 | RANDOM_HALF | 667 | 624 | 43 | 0.005333108 | -0.005447249 |
| 5 | 8000 | 2 | PROTO_HALF | 667 | 665 | 2 | 0.006134129 | -0.005836367 |
| 6 | 8000 | 8 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 6 | 8000 | 8 | EMA_ALL | 4269 | 3913 | 356 | 0.005376460 | -0.005472236 |
| 6 | 8000 | 8 | EMA_HALF | 2129 | 2000 | 129 | 0.002854900 | -0.002688577 |
| 6 | 8000 | 8 | RANDOM_HALF | 2129 | 1949 | 180 | 0.002672100 | -0.002723011 |
| 6 | 8000 | 8 | PROTO_HALF | 2129 | 2004 | 125 | 0.002862965 | -0.002703084 |
| 7 | 8000 | 8 | BASE | 0 | 0 | 0 | 0.000000000 | 0.000000000 |
| 7 | 8000 | 8 | EMA_ALL | 2490 | 2235 | 255 | 0.004185876 | -0.004202257 |
| 7 | 8000 | 8 | EMA_HALF | 1242 | 1164 | 78 | 0.002337497 | -0.002085582 |
| 7 | 8000 | 8 | RANDOM_HALF | 1242 | 1110 | 132 | 0.002067266 | -0.002071542 |
| 7 | 8000 | 8 | PROTO_HALF | 1242 | 1153 | 89 | 0.002334794 | -0.002126579 |

## Interpretation and boundaries

This diagnostic resolves whether a fully specified independent EMA-only location/budget rule is useful on the current development images and whether prototype ranking adds to simple matched-budget alternatives. It does not validate an unseen-image selector. All8ENTRY100 contexts,40current A_fit image/context exposures,canonicalphoto transforms and computationalseed119 were deliberately reused; models already trained on these images. Prototype support excludes only its current target image(n=2 one other support,n=8 two). Repeated image/condition/pixel exposures are not independent samples, and n-group differences cannot be attributed solely to labeling efficiency. V119–V122 are successive development diagnostics, not four independent confirmations.

Action11's mixture is memory-dominated whenever its memory gate is active. The current interventions resolve selected EMA-memory disagreements toward EMA, retaining/permuting original blended probabilities. Even a positive diagnostic cannot establish that memory should be globally weakened or removed; the current EMA saw these labels and old-task protection was not evaluated. True-label NLL is different from native KL(target||student), and weighted target L1 is not a measured parameter-gradient norm.

The full-rule arm and half-budget rankings answer different questions. A stronger full-rule effect can arise from more coverage; it must not be called superior per-location ranking. All HALF methods have equal count/destination/weight, but continuous probability perturbations may differ. One pre-registered random permutation is one realization, not an estimated randomized expectation. Prototype ranking reuses a fixed scoring definition chosen before this readout; no tuning after results.

Before an unseen-image claim, separately freeze a legal training-only model-level holdout excluding target images from relevant fitting. Removing images from prototype support after ENTRY100 is not enough; n=2 leave-one-fit would have only1fitted label and must be reported accordingly. Do not access sealed Q_dev/test/hidden U labels. Before training-value claims, use a separate same-entry/stream/budget comparison and examine new-task gain plus old-task retention. No fullGRPO, segmentation training, new history bank or next experiment was launched in this closeout. History-free trained-parameter memory remains unchanged.


## Main empirical finding

The primary independent rule passes all frozen sign conditions without any prototype-derived locations or per-image quotas: EMA_ALL acts on15791exposures, repairs14272 and harms1519, giving equal-context weighted accuracy+0.005896417590(+0.589642percentage points) and NLL−0.005664188041, both positive directions in8/8contexts and all source/size subgroups. This supports the simple conflict rule on these development images. It does not establish safety on U or old-task retention. The full CE count exceeds V121's12774active shuffle pool because the prior pool omitted strata with no prototype candidates; the two totals were not intended to be identical.

All three HALF arms use7874interventions and weighted changed mass5452.5. EMA_HALF repairs7290/harm584; RANDOM_HALF7118/756; PROTO_HALF7329/545. All improve both metrics versus BASE in8/8contexts, including the now prototype-independent RANDOM_HALF. However, EMA_HALF minus RANDOM_HALF has worse mean NLL(+0.000113841715) and only2/8joint-positive contexts. PROTO_HALF minus RANDOM_HALF also has worse mean NLL(+0.000056629613) and only3/8joint-positive contexts. Both ranking increments therefore fail.

PROTO_HALF minus EMA_HALF accuracy is only+0.000018769967(+0.001877percentage points), NLL−0.000057212102, with3/8joint-positive contexts and a negative source8000 mean accuracy increment. The masks overlap7111/7874positions(about90.3%). Thus there is no reliable prototype ranking increment over the simple EMA margin at this fixed budget, despite a slightly favorable overall average. Do not advertise the candidate with the numerically largest half-arm accuracy as a validated winner.

EMA_ALL has the largest absolute target-quality effect here but more than twice the half-arm intervention count; this is not evidence that its per-position ranking is better. The main supported development finding is usefulness of these specific observable conflict directions, not a necessity for confidence/prototype sorting. Next evidence must address actual model-level holdout and then paired new/old-task training, rather than retuning fractions or ranking scores on the same eight conditions. No such additional run was executed here.

## Costs and public delivery

New40current-labeled image/context reads/evaluations,160model-image forwards(40eachEMA/memory/flippedmemory/student),40prototype assignments shared acrossfive diagnostic arms. V119–V122combined640forwards160labeled evaluations, all separately charged. Zero native/actor optimizer,linear solves,Q_train/Q_dev,U-images,hidden labels or new annotations. Cumulative841061native excluding8000source(849061including),181593actor,57solves unchanged; historical failures and512/146CPUactor-forward diagnostics retained. Synthetic selfcheck/audit/report add0model/image/optimizer. A deployed EMA-only rule needs no prototype/support-label computation; shared diagnostic inference is not a standalone latency comparison.

Public: source/protocol/selfcheck,all40anonymous metric rows,8control summaries,all6comparisons and contexts/subgroups,support/reference/qualification/cost/execution/audit/report. Individual image metrics/support IDs/features/data/weights/rawlogs remainprivateNAS. Final committed-file NAS delivery,proxyGitHub push/remote branch/anonymous report verification recorded in FINAL_PUBLICATION.json.
