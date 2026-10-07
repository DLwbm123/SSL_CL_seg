# V92 current-policy values

STOP_V92_NO_CURRENT_POLICY_INCREMENTAL_GAIN

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

Fixed original deployment utility. Same denseCE initialization/objective/fit budget as stale-value controls; current versus old behavior collection. All104 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Result and attribution

The fixed primary failed. Fresh-policy RL reaches74.023156% new and81.282564% old foreground soft Dice, versus freshCE74.061105%/81.266149%. Its utility is-0.068442344: -0.000245071 relative to freshCE, -0.001040004 relative to stale alignedRL, and-0.002579086 relative to stale continuedCE. Both RL seeds lose to their stale-policy counterparts (-0.002056694/-0.000023313); versus freshCE they split (+0.000803956/-0.001294097). The required pairwise and practical-gain tests fail. The positive uniform delta(+0.247889 percentage points new,-0.006503points old,+0.001889113utility) does not replace the primary.

FreshCE also loses to stale continuedCE by0.002334015utility. Thus this finite current-policy refresh does not support the proposed improvement; it does not prove that policy/value mismatch never matters. Same initial actor weights, objectives and1024fitsteps are used for fresh/stale counterparts; only collecting behavior and resulting MonteCarlo states/values differ. The collecting policy is a fixed two-seed probability average, not a moving actor. This remains one full-action offline policy-improvement experiment, not GRPO or iterative online RL.

All104endpoints and per-context/channel/seed comparisons are preserved. Four shared development contexts, two actor seeds and common student streams are not independent student or patient replicates. CurrentD1 development has been observed repeatedly. Old means simulated retention withinD1. Foreground scores are macro rim/cup soft Dice. No independent generalization or paper-ready positive claim is supported. No run/seed/checkpoint was discarded.

## Verification and costs

All26jobs exited0; coordinator exited and lockreleased. The104endpoint seal precedes allnew evaluation jobs, and the frozenprimary recomputes exactly. Newphysicalattempts=success:8nativequalification+41600collection+3200development=44808student updates;4096actor updates;zero failures. All16same-trajectory continuation reuses have receipts. Newtraining queryimageevaluations2752, development192, no newuniqueimages orhidden Ulabels. Existing labelroles remain16memory-fit, support2/8,4old+4newtrainingquery,4old+4newdevelopmentquery; repeatedqueries mustnotbe mistakenfor additionalindependentimages/patients.

Bothold andnew complete-actioncollection cost41600studentupdates. Stale results were reused physically; methods depending on those data must still be attributed their generation cost. Querysampling differs because the old run also collected local-horizon diagnostics; there is no claim of identical walltime orFLOPs. CumulativecampaignthroughV92:329693studentupdates excluding8000commonsource(337693includingit),62669actorupdates. Historicalpreparation andfailedstagecostsremain counted. Privateimages/labels/states/checkpoints remainNAS. OriginalSTOPs andV84NOT_RUN unchanged.

Before another optimizer intervention, a separately preregistered zero-update audit will examine action-value repeatability across the two existing streams and saved policy concentration. This is a new diagnostic, not a retrospective change to the V92 criterion.
