# V91 aligned KL reference

STOP_V91_NO_ALIGNED_ANCHOR_INCREMENTAL_GAIN

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
| Z_1024 | 0.739401684 | 0.811838625 | -0.070147864 |
| LOCAL_CE | 0.739543077 | 0.812147744 | -0.069772382 |
| EPISODIC_CE | 0.737805664 | 0.811187826 | -0.071929588 |
| EPISODIC_RL | 0.738702079 | 0.811807885 | -0.070549847 |
| DENSE_CE | 0.741447121 | 0.813895322 | -0.066159520 |
| DENSE_RL | 0.738687845 | 0.811814298 | -0.070565288 |
| CONTINUE_CE | 0.741702477 | 0.813926101 | -0.065863258 |
| OLD_ANCHOR_RL | 0.740043642 | 0.812704927 | -0.068703993 |
| ALIGNED_ANCHOR_RL | 0.740760739 | 0.813344812 | -0.067402340 |

Fixed original deployment utility. Same denseCE initial weights for all three arms; old vs aligned KL reference and continuedCE controls. All88 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.

## Interpretation and attribution

The primary remains STOP. Aligned-reference RL achieves74.076074% new and81.334481% old foreground soft Dice, versus uniform73.775267%/81.289067% and continuedCE74.170248%/81.392610%. Relative to uniform, alignedRL gains0.300807 percentage points new and0.045415 points old, utility+0.002929116; relative to matched continuedCE it loses0.094174 points new,0.058129 points old and0.001539082 utility. The two seed deltas against continuedCE are+0.000218020 and-0.003296184, so the required seedwise condition also fails. Neither positive uniform comparison nor training-table success replaces the preregistered primary.

Changing only the KL reference with matched initialization/fit budget improves utility over old-reference RL by0.001301653 on average; both seed differences are positive (+0.002185435/+0.000417870). This supports an effect of the reference under this finite development recipe. It does not establish superiority to continuedCE or independent generalization. The earlier training-objective diagnosis remains valid on the fixed table; it did not guarantee a deployment gain.

An exploratory paired trace localizes the largest loss todev2/seed602: alignedRL actions[8,8] versus continuedCE[2,8], with utility difference-0.016251811. Seed601 uses[2,8] for both and has identical endpoints there. Other contexts include positive and negative differences, all retained in CONTEXT_COMPARISONS and ENDPOINT_RESULTS. Actor seeds share the student stream and private categorical draws; they are not independent student replications. No seed is dropped.

The next unresolved issue is off-policy value/state collection: allV89/V90 reward branches were collected with the older V86 behavior policy, while V91's current/reference policy is V90denseCE. At step100, continuation values therefore refer to the older policy; at step200, retained states were visited by that older behavior. This is a verified protocol mismatch and a hypothesis for performance limits, not an established causal explanation of thedev2 loss. A separate iteration must freeze a matched stale/fresh collection comparison before new rollouts.

## Verification and costs

All10 jobs exited0. All88 endpoints were sealed before the four new evaluation jobs started; the primary was recomputed exactly. Physical attempts=success:6144actor,8native qualification,4800native development, with zero failed calls. There were no new training-query accesses and288development image evaluations on the same eight existing query images. No new unique images, role permissions or hidden U labels. Memory-fit16/support2or8/training-query4old+4new/development-query4old+4new label budgets remain disclosed; repeated role access is not an independent patient cohort. Old means simulated retention withinD1, foreground metric averagesrim/cup soft Dice.

Cumulative throughV91:284885student updates excluding8000common source (292885including it),58573actor updates. The shared V90CE initializer cost1024updates perseed is historical and not free; each newarm adds1024updates perseed. The old-reference warm-start comparison withV90RL additionally changes initialization and CEpreparation cost, so it is exploratory. Private data/states/checkpoints remain onNAS. EarlierSTOPs andV84NOT_RUN remain unchanged. No claim of paper-ready independent positive evidence.

GitHubHTTP500 affected the initial publication attempt only; startup publication recovered on the scheduled proxy retry. No experiment was retried, reinitialized or extended.
