# V93 fixed actor representation

STOP_V93_NO_FIXED_REPRESENTATION_INCREMENTAL_GAIN

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
| CONTINUE_CE_601 | 0.741826036 | 0.813868878 | -0.065827812 |
| CONTINUE_CE_602 | 0.741578919 | 0.813983323 | -0.065898705 |
| OLD_ANCHOR_RL_601 | 0.740455020 | 0.813191734 | -0.067795227 |
| OLD_ANCHOR_RL_602 | 0.739632264 | 0.812218120 | -0.069612759 |
| ALIGNED_ANCHOR_RL_601 | 0.741558542 | 0.814318964 | -0.065609792 |
| ALIGNED_ANCHOR_RL_602 | 0.739962937 | 0.812370660 | -0.069194889 |
| FRESH_CE_601 | 0.740409646 | 0.812550550 | -0.068470442 |
| FRESH_CE_602 | 0.740812449 | 0.812772429 | -0.067924105 |
| FRESH_RL_601 | 0.740515739 | 0.813274937 | -0.067666486 |
| FRESH_RL_602 | 0.739947379 | 0.812376341 | -0.069218202 |
| HEAD_CE_601 | 0.740449224 | 0.812483527 | -0.068499546 |
| HEAD_CE_602 | 0.739603795 | 0.812230946 | -0.069643640 |
| HEAD_RL_601 | 0.740555316 | 0.813207913 | -0.067695590 |
| HEAD_RL_602 | 0.739947379 | 0.812376341 | -0.069218202 |
| Z_1024 | 0.739401684 | 0.811838625 | -0.070147864 |
| LOCAL_CE | 0.739543077 | 0.812147744 | -0.069772382 |
| EPISODIC_CE | 0.737805664 | 0.811187826 | -0.071929588 |
| EPISODIC_RL | 0.738702079 | 0.811807885 | -0.070549847 |
| DENSE_CE | 0.741447121 | 0.813895322 | -0.066159520 |
| DENSE_RL | 0.738687845 | 0.811814298 | -0.070565288 |
| CONTINUE_CE | 0.741702477 | 0.813926101 | -0.065863258 |
| OLD_ANCHOR_RL | 0.740043642 | 0.812704927 | -0.068703993 |
| ALIGNED_ANCHOR_RL | 0.740760739 | 0.813344812 | -0.067402340 |
| FRESH_CE | 0.740611047 | 0.812661489 | -0.068197274 |
| FRESH_RL | 0.740231559 | 0.812825639 | -0.068442344 |
| HEAD_CE | 0.740026509 | 0.812357237 | -0.069071593 |
| HEAD_RL | 0.740251347 | 0.812792127 | -0.068456896 |

Fixed original deployment utility. Same denseCE initialization/objective/fit budget as stale-value controls; head-only versus full-network actor updates. All120 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Outcome and matched comparisons

The primary failed. HEAD_RL reaches 74.025135% new and 81.279213% old foreground macro rim/cup soft Dice, utility -0.068456896. Relative to full-network FRESH_RL, utility changes by -0.000014552 (seed 601: -0.000029104; seed 602: exactly zero). Freezing the representation does not improve deployment under this fixed recipe. This does not establish that representation learning can never matter.

HEAD_RL exceeds HEAD_CE by +0.000614697 utility, but HEAD_CE itself deteriorates relative to full-network FRESH_CE. HEAD_RL remains below FRESH_CE (-0.000259623), aligned stale-value RL (-0.001054556), and continued CE (-0.002593638). Its +0.001874561 utility relative to uniform does not replace these failed comparisons. The practical tradeoff passes against uniform only; it fails against continued CE, fresh CE and head CE. Both seeds and all 120 endpoints are retained in the tables, including new/old channel scores and per-context comparisons. Absolute forgetting is 0.050295084 against the frozen memory reference.

## Preregistered descriptive diagnostics

The zero-query, zero-update audit completed before all new actor fits. Cross-stream best-action agreement is 14/16 (87.5%) for V90 and 13/16 (81.25%) for V92; mean cross-stream raw regret is 0.000284212 and 0.000375432 respectively. None of the paired state vectors are identical, including step 100: provider streams and feature acquisition can differ despite shared student entry weights. Thus these are context/time cross-stream robustness diagnostics, not pure identical-state Monte Carlo noise estimates. Most choices agree, but mismatches include large relative regrets. All pairs, directions and states are reported.

V92 full-network RL has training entropy about 0.634 versus CE 0.929. At the four shared development entry states, mean between-seed total variation is 0.364441 for RL versus 0.179734 for CE. All 32 saved categorical choices replay exactly with the original private action RNG. Hidden-layer changes account for approximately 0.33–0.43 of the sum of hidden/head centered-logit magnitudes for RL across training and entry development states. This is an algebraic magnitude decomposition with possible cancellation, not a causal fraction of performance. The new frozen-hidden intervention nevertheless gives no deployment improvement. Head-only RL still obtains higher training return than head-only CE while failing the overall development criterion.

## Verification, costs and limits

All 10 jobs exited zero, the coordinator ended and its lock is free. All 120 endpoints (104 audited reuse, 16 new) were sealed before every new evaluation job started. The primary recomputes exactly from complete endpoint rows. Frozen 800 hidden parameters remain unchanged and receive no gradients; all 297 head parameters are the optimizer's intended subset, with nonzero head gradient verified. New costs: 4096 actor updates, 8 qualification plus 3200 development student updates, zero failed updates, zero new training-query evaluations and 192 development image-evaluations. No new unique images or labels. Cumulative campaign: 332901 student updates excluding 8000 common source updates (340901 including them), 66765 actor updates. The reused V92 table still requires attribution of its 41600 student collection updates and 2752 training image-evaluations; reuse is not free information generation.

Original label roles remain 16 memory-fit images, support 2/8, four old and four new training-query images, and four old and four new development-query images. Repeated image-evaluations are not new independent images. Current old retention is simulated within D1. Four shared contexts and two actor seeds do not establish patient independence or independent student replication. D1 development has been repeatedly observed; no independent confirmation or paper-ready positive claim is supported. All previous STOP decisions and V84 NOT_RUN remain. Private images, labels, role identities, state tensors and weights stay on NAS.
