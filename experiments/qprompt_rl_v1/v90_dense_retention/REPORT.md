# V90 dense retention credit

STOP_V90_NO_DENSE_RETENTION_GAIN

| Method | New | Old | Utility |
|---|---:|---:|---:|
| NATIVE | 0.734741107 | 0.812819198 | -0.073255628 |
| FIXED_BEST | 0.737623854 | 0.810925372 | -0.072268197 |
| UNIFORM_ACTION | 0.737752672 | 0.812890666 | -0.070331457 |
| TIME_FIXED | 0.737614123 | 0.811875904 | -0.072033168 |
| Z_1024_601 | 0.739587104 | 0.812129237 | -0.069718187 |
| Z_1024_602 | 0.739216264 | 0.811548013 | -0.070577540 |
| LOCAL_CE_601 | 0.739543077 | 0.812147744 | -0.069772382 |
| LOCAL_CE_602 | 0.739543077 | 0.812147744 | -0.069772382 |
| EPISODIC_CE_601 | 0.737905312 | 0.811192308 | -0.071846252 |
| EPISODIC_CE_602 | 0.737706015 | 0.811183345 | -0.072012924 |
| EPISODIC_RL_601 | 0.738736102 | 0.811767960 | -0.070548958 |
| EPISODIC_RL_602 | 0.738668056 | 0.811847810 | -0.070550735 |
| DENSE_CE_601 | 0.741068206 | 0.813921766 | -0.066491228 |
| DENSE_CE_602 | 0.741826036 | 0.813868878 | -0.065827812 |
| DENSE_RL_601 | 0.738707634 | 0.811780786 | -0.070579840 |
| DENSE_RL_602 | 0.738668056 | 0.811847810 | -0.070550735 |
| Z_1024 | 0.739401684 | 0.811838625 | -0.070147864 |
| LOCAL_CE | 0.739543077 | 0.812147744 | -0.069772382 |
| EPISODIC_CE | 0.737805664 | 0.811187826 | -0.071929588 |
| EPISODIC_RL | 0.738702079 | 0.811807885 | -0.070549847 |
| DENSE_CE | 0.741447121 | 0.813895322 | -0.066159520 |
| DENSE_RL | 0.738687845 | 0.811814298 | -0.070565288 |

Fixed original deployment utility; no metric replacement. Signed-retention reward is training-only. All64 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Interpretation

The preregistered RL primary failed. DENSE_RL new/old foreground soft Dice is 73.868784%/81.181430%; DENSE_CE is 74.144712%/81.389532%. Relative to uniform, DENSE_CE improves new by0.369445 percentage points and old by0.100466 points, utility by0.004171937. Both CE seeds have positive utility improvements over uniform (+0.003840229/+0.004503645). These are secondary, exploratory gains for CE; they do not replace the failed RL primary or establish independent confirmation.

DENSE_RL barely changes from the original episodic RL: utility -0.000015441 relative to it, versus -0.004405768 relative to DENSE_CE. Dense retention rewards therefore provide a promising signal for this CE recipe, while the existing RL objective does not exploit it in deployment. The signed old-score intervention improved CE over identical-data episodic CE by0.005770068 utility. Reward and objective effects must be kept separate.

The old-action range median is0.004955374, exceeding the preregistered0.0001 signal threshold. This establishes measured action-dependent old-score variation on training queries, not a generalization claim. All64 endpoints, both actor seeds, per-channel readouts and paired context comparisons remain in the public tables. Actor seeds share student trajectories and categorical random draws; they are not independent student replicates. The old domain is simulated within currentD1. Foreground scores average rim/cup soft Dice; no patient-independent or real later-domain claim is supported.

A next hypothesis is that KL to the old local-reward actor constrains movement toward the new retention-aware policy. That hypothesis is not proven by these endpoint results; it needs matched initialization and controls before attributing the gap to the anchor. No coefficient, seed, or checkpoint was selected on development scores.

## Validation and costs

All18 jobs exited0, primary gate recomputed exactly, and four evaluation jobs began after64endpoint lock. Counts:3208 successful native updates (8qualification+3200development),4096actor updates, zero failures;1120training-query image evaluations and192development-query evaluations. No new unique images or role changes. Existing budgets remain16memory-fit images, support2/8, eight training-query images and eight development-query images with repeated access; these role counts are not an independent patient cohort. Hidden U labels remain inaccessible.

Cumulative throughV90:280077student updates excluding8000common source (288077including it),52429actor updates. Historical budgets/STOPs are preserved, V84 remains NOT_RUN. Private data, checkpoints and state vectors remainNAS. This stage completes normally despite the negative primary.
