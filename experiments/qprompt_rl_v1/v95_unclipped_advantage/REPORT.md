# V95 unclipped pooled advantage

STOP_V95_NO_UNCLIPPED_ADVANTAGE_INCREMENTAL_GAIN

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
| UNCLIPPED_CE_601 | 0.739497703 | 0.811506560 | -0.070447596 |
| UNCLIPPED_CE_602 | 0.741550499 | 0.813643634 | -0.066251473 |
| UNCLIPPED_RL_601 | 0.741518964 | 0.814385988 | -0.065580688 |
| UNCLIPPED_RL_602 | 0.741547433 | 0.814373162 | -0.065549806 |
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
| UNCLIPPED_CE | 0.740524101 | 0.812575097 | -0.068349534 |
| UNCLIPPED_RL | 0.741533198 | 0.814379575 | -0.065565247 |

Fixed original deployment utility. Same denseCE initialization/objective/fit budget as V94; unclipped versus clipped pooled advantages. All152 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Outcome and exact endpoint comparison

The primary is STOP. All 16 newly trained development endpoints have exactly the same saved action sequences, new-short/new-final/old-final channel scores, gain, forgetting and utility as their matched V94 clipped counterparts. This is a verified equality of saved metrics and actions, not a file-byte or checkpoint identity claim. Neither RL seed improves over POOLED_RL; both paired utility changes are zero. UNCLIPPED_RL retains 74.153320% new /81.437957% old Dice and utility -0.065565247. Its gain versus continued CE remains only +0.000298011, with the same failed practical tradeoffs. The advantage-truncation intervention does not improve these fixed deployment trajectories.

This is an executed negative ablation, not reuse relabeled as new training: 3200 development student updates and 192 new query image-evaluations were incurred. Equal sampled actions can yield equal native student endpoints even when actor probabilities differ. Both seeds, all contexts and the previous variants remain in the results; no favorable seed or trajectory is selected.

## Training diagnostics

The pooled RMS matches V94 exactly at 0.0026099849492311478. Removing truncation retains a maximum absolute normalized advantage of 4.033438 and affects three of288 values. Training raw return and entropy change only slightly: RL raw dense returns 0.004283082/0.004282350, entropy 0.698443/0.697628; CE raw returns 0.004075292/0.004075880 and entropy 1.226287/1.225811. These do not establish meaningful optimization improvement. The full per-state extrema, affected-action counts and fit logs are preserved. Gradient norm clipping remains1 and all losses/gradients were finite; only advantage-value clipping was removed.

V94's paired improvement over per-state scaling remains a limited development observation. V95 supplies no incremental gain. The previous cross-stream audit found differing state vectors and some action-gap/ranking disagreements within matched context/time pairs. A subsequent independently registered target-pooling experiment can test invariance across those paired training trajectories, but cannot assume the two states are identical or call their differences pure label noise.

## Verification, costs and limits

All10jobs exited zero; coordinator exited and lock released. All152 endpoints (136 reused plus16new) sealed before all four new evaluation jobs. The frozen primary recomputes exactly from complete rows. Attempts equal success:4096actor and3208native updates (8qualification+3200development), zero failed updates, zero new training queries,192development image-evaluations, no new unique images. Native paired restore and inference/RNG isolation pass. Cumulativecampaign:339317native excluding8000common source (347317including),74957actor updates. The reused V92 table retains its attributed generation cost of41600native and2752training image-evaluations.

Original label roles remain16memory-fit images, support2/8,4old+4new training-query and4old+4new development-query images. Repeated evaluations do not add independent images. Old retention is simulated withinD1; the development set is repeatedly observed. Four shared contexts and two actor seeds are not independent patients or student replications. No independent confirmation or paper-ready positive claim is supported. All previous STOPs andV84 NOT_RUN remain. Private data, role identities, states and weights stay onNAS.
