# V94 pooled advantage scale

STOP_V94_NO_POOLED_ADVANTAGE_INCREMENTAL_GAIN

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
| POOLED_CE_601 | 0.739497703 | 0.811506560 | -0.070447596 |
| POOLED_CE_602 | 0.741550499 | 0.813643634 | -0.066251473 |
| POOLED_RL_601 | 0.741518964 | 0.814385988 | -0.065580688 |
| POOLED_RL_602 | 0.741547433 | 0.814373162 | -0.065549806 |
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
| POOLED_CE | 0.740524101 | 0.812575097 | -0.068349534 |
| POOLED_RL | 0.741533198 | 0.814379575 | -0.065565247 |

Fixed original deployment utility. Same denseCE initialization/objective/fit budget as V92; pooled versus per-state advantage scale. All136 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Outcome and matched attribution

The primary remains STOP. POOLED_RL reaches 74.153320% new and 81.437957% old foreground macro rim/cup soft Dice, utility -0.065565247. Both actor seeds beat the matched pooled CE (+0.004866909/+0.000701666 utility) and previous per-state FRESH_RL (+0.002085798/+0.003668396). Mean utility gains are +0.002784287 over pooled CE and +0.002877097 over FRESH_RL. Thus this fixed intervention improves the paired development outcome for both actor initializations, but does not establish independent generalization.

Against the strongest historical continued CE, utility increases by only +0.000298011, below the preregistered +0.0005 requirement. New Dice is -0.016928 percentage points and old Dice +0.045347 points relative to that control. The practical tradeoff fails against continued CE, fresh CE and pooled CE; only the uniform tradeoff passes. The +0.378053/+0.148891 points new/old versus uniform cannot replace the failed primary. Absolute forgetting against the frozen memory reference remains 0.048707636.

The action traces localize the FRESH_RL improvement: at dev2 both pooled RL seeds choose [2,8], replacing [5,8]/[8,8]. Dev0 and dev1 choices remain identical to each seed's FRESH_RL choices. At dev3 seed601 stays [0,6], while seed602 changes [0,8] to [0,6], reducing utility there. This is descriptive analysis of already-observed outcomes, not permission to hardcode a context action, choose a seed or change deployment RNG. All contexts and seeds remain in the comparison.

## Training scale diagnostics

The training-only pooled RMS is 0.0026099849492311478. Old floored per-state scales divided by it range from 0.282736 to 2.324904. The new normalization therefore lowers relative action advantages in low-gap states while raising them in high-gap states. Three of 288 centered scaled values still hit the original magnitude-3 clip, all at the m8000_n2_contrast step100 context (one in stream1, two in stream2). This is an observed remaining nonlinearity, not proof that removing it improves deployment.

POOLED_RL training entropy is 0.698441/0.697628, versus 1.226287/1.225811 for pooled CE. Its training raw dense return is 0.004283083/0.004282348 versus pooled CE 0.004075293/0.004075880. Reference KL is about 0.088 for RL and 0.288 for CE. Comparing normalized-return numbers across V92 and V94 is not meaningful because the normalization changed. These results support a limited within-development scaling effect; they do not establish that policy entropy or raw reward alone explains the outcome. All diagnostic rows and fit logs are preserved.

## Verification, costs and limits

All 10 jobs exited zero; the coordinator ended and its lock is free. All 136 endpoints (120 reused, 16 new) sealed before every new evaluation job. The complete primary recomputes exactly. Physical attempts equal successful updates: 4096 actor, 8 native qualification and 3200 native development; zero failures, zero new training-query evaluations, 192 development image-evaluations, no new unique images. Scale-shift/global-amplitude invariance and finite-gradient checks passed. Cumulative campaign: 336109 student updates excluding 8000 common source updates (344109 including them), 70861 actor updates. Reused reward data still require attribution of 41600 native collection updates and 2752 training image-evaluations.

Label roles remain 16 memory-fit images, support 2/8, four old plus four new training-query images, and four old plus four new development-query images. Repeated queries are not independent images. Old retention is simulated within D1. Four shared development contexts and two actor seeds do not establish patient independence or independent student replication. D1 has been repeatedly observed; this is not independent confirmation or paper-ready positive evidence. All original STOPs and V84 NOT_RUN remain. Private images, labels, role identities, states and weights remain on NAS.
