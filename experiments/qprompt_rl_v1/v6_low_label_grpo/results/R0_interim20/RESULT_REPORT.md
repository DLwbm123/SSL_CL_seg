# R0: 20% supervision interim results

User-authorized early evaluation on 2026-10-05. All 18 registered endpoints in this scope and all 42 preparation/training jobs completed. The 10%/5% matrix remains in training; these scores do not establish a low-label advantage or pass either success gate. Original full-matrix-before-val timing was amended before scoring; remaining settings and thresholds stay fixed.

One fixed REFUGE source/segmentation seed 168, label subset 5101, controller 401; no independent replication. RIM uses 16/79 labeled training images (20.25%), Drishti 10/51 (19.61%); one selected label per domain is included in the controller reward budget. Historical development validation is reused. ORIGINAL means target-domain supervised adaptation with lambda_U=0, not the untouched source model.

Macro Dice averages rim and cup; disc-union is reported separately. Scores below are percentages, differences are percentage points.

The 20% results do not show a useful GRPO advantage: G2/G4/G8 trail the best non-RL method selected within each domain by 0.8161/0.8039/0.8263pp on average. Their gains over matched PPO are only 0.0057/0.0225/0.0000pp. GRPO retains 1.3162–1.4810pp more old-domain Dice than ORIGINAL on average, but still loses 10.7404–14.3558pp relative to the fixed source. The main adverse new-domain result is RIM, where all GRPO settings reduce both rim and cup Dice versus FINE_05. This single run provides no confidence interval or replication claim; 10%/5% outcomes remain unknown.

## New-domain results

| Method | RIM macro | Drishti macro | Equal-domain mean |
|---|---:|---:|---:|
| ORIGINAL | 71.1214 | 74.9987 | 73.0601 |
| FINE_05 | 71.1937 | 74.5724 | 72.8831 |
| OFFLINE_25 | 69.5887 | 74.9653 | 72.2770 |
| GRPO_FS_G2 | 69.5642 | 74.9961 | 72.2801 |
| PPO_MATCHED_G2 | 69.5835 | 74.9653 | 72.2744 |
| GRPO_FS_G4 | 69.5887 | 74.9961 | 72.2924 |
| PPO_MATCHED_G4 | 69.5745 | 74.9653 | 72.2699 |
| GRPO_FS_G8 | 69.5745 | 74.9653 | 72.2699 |
| PPO_MATCHED_G8 | 69.5745 | 74.9653 | 72.2699 |

## GRPO paired differences

| G | vs best non-RL per domain | vs same-G PPO | old REFUGE vs ORIGINAL |
|---|---:|---:|---:|
| 2 | -0.8161 | +0.0057 | +1.4810 |
| 4 | -0.8039 | +0.0225 | +1.3475 |
| 8 | -0.8263 | +0.0000 | +1.3162 |

Per-domain comparisons and all rim/cup/disc changes are retained in DECISION.json. No candidate is selected from these 20% results.

## Every channel and old-domain retention

| Target | Method | Rim | Cup | Disc-union | Old REFUGE | Old change from source (pp) |
|---|---|---:|---:|---:|---:|---:|
| RIM_ONE_r3 | ORIGINAL | 74.1765 | 68.0663 | 89.2188 | 71.3164 | -11.8088 |
| RIM_ONE_r3 | FINE_05 | 74.1326 | 68.2549 | 89.1718 | 73.7041 | -9.4211 |
| RIM_ONE_r3 | OFFLINE_25 | 73.1110 | 66.0664 | 88.4953 | 72.1177 | -11.0075 |
| RIM_ONE_r3 | GRPO_FS_G2 | 73.1118 | 66.0165 | 88.4333 | 72.3848 | -10.7404 |
| RIM_ONE_r3 | PPO_MATCHED_G2 | 73.1080 | 66.0589 | 88.4872 | 72.1344 | -10.9908 |
| RIM_ONE_r3 | GRPO_FS_G4 | 73.1110 | 66.0664 | 88.4953 | 72.1177 | -11.0075 |
| RIM_ONE_r3 | PPO_MATCHED_G4 | 73.1007 | 66.0484 | 88.4892 | 72.1349 | -10.9903 |
| RIM_ONE_r3 | GRPO_FS_G8 | 73.1007 | 66.0484 | 88.4892 | 72.1349 | -10.9903 |
| RIM_ONE_r3 | PPO_MATCHED_G8 | 73.1007 | 66.0484 | 88.4892 | 72.1349 | -10.9903 |
| Drishti_GS | ORIGINAL | 73.4051 | 76.5924 | 93.1183 | 66.9554 | -16.1698 |
| Drishti_GS | FINE_05 | 72.7470 | 76.3977 | 94.1526 | 68.6222 | -14.5030 |
| Drishti_GS | OFFLINE_25 | 73.1818 | 76.7488 | 93.5265 | 68.7694 | -14.3558 |
| Drishti_GS | GRPO_FS_G2 | 73.2089 | 76.7832 | 93.4973 | 68.8491 | -14.2762 |
| Drishti_GS | PPO_MATCHED_G2 | 73.1818 | 76.7488 | 93.5265 | 68.7694 | -14.3558 |
| Drishti_GS | GRPO_FS_G4 | 73.2089 | 76.7832 | 93.4973 | 68.8491 | -14.2762 |
| Drishti_GS | PPO_MATCHED_G4 | 73.1818 | 76.7488 | 93.5265 | 68.7694 | -14.3558 |
| Drishti_GS | GRPO_FS_G8 | 73.1818 | 76.7488 | 93.5265 | 68.7694 | -14.3558 |
| Drishti_GS | PPO_MATCHED_G8 | 73.1818 | 76.7488 | 93.5265 | 68.7694 | -14.3558 |

## Denominators and costs

Validation image counts (no patient identifiers): `{"RIM_ONE_r3": {"REFUGE": 100, "RIM_ONE_r3": 40}, "Drishti_GS": {"REFUGE": 100, "Drishti_GS": 25}}`. Means are over images within domain, then equally over the two domains.

20% scope consumed 347166 student updates. Detailed optimizer calls are in TRAINING_COSTS.json. Early evaluation adds zero student updates and reuses the registered jobs; all 36 entry/endpoint evaluator receipts are in COSTS.json. These resources are part of the main campaign totals, not additional training or a new candidate.

## Policy/reward diagnostics

| Domain | Variant | Final panel argmax change vs initialization (%) | Final KL vs initialization | Mean within-group reward std |
|---|---|---:|---:|---:|
| RIM_ONE_r3 | GRPO_FS_G2 | 0.0000 | 0.00198600 | 0.19313373 |
| RIM_ONE_r3 | PPO_MATCHED_G2 | 0.0000 | 0.00043838 | 0.23947625 |
| RIM_ONE_r3 | GRPO_FS_G4 | 0.0000 | 0.00058842 | 0.18439358 |
| RIM_ONE_r3 | PPO_MATCHED_G4 | 0.0000 | 0.00063136 | 0.26116416 |
| RIM_ONE_r3 | GRPO_FS_G8 | 0.0000 | 0.00156789 | 0.30084886 |
| RIM_ONE_r3 | PPO_MATCHED_G8 | 0.0000 | 0.00075532 | 0.30084886 |
| Drishti_GS | GRPO_FS_G2 | 0.0000 | 0.00244285 | 0.02508930 |
| Drishti_GS | PPO_MATCHED_G2 | 0.0000 | 0.00221609 | 0.02027968 |
| Drishti_GS | GRPO_FS_G4 | 0.0000 | 0.00077242 | 0.03676291 |
| Drishti_GS | PPO_MATCHED_G4 | 0.0000 | 0.00052076 | 0.03362464 |
| Drishti_GS | GRPO_FS_G8 | 0.0000 | 0.00066586 | 0.02400468 |
| Drishti_GS | PPO_MATCHED_G8 | 0.0000 | 0.00127789 | 0.02400468 |

Panel argmax agreement is measured on fixed initialization contexts; it does not prove identical action sequences on learner-dependent endpoint states. Single-label online reward can overfit. Full group diagnostics and endpoint action counts are retained; no unfavorable domain/channel is removed.

Execution source: `51caea6359fd196962d21b8c85e231a37f52b87d`; early evaluation runner/amendment: `f38d4007035e7e9cefb086b274d39b9a9b6349eb`. The original healthy coordinator was not restarted. At campaign completion regenerate the report with the amended audit function before publication.
