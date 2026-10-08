# V101 current expanded-policy returns

**NO_V101_CURRENT_EXPANDED_RL_INCREMENTAL_GAIN.** Completed 2026-10-08T12:54:12.059009+00:00. Refreshing training returns and reached states under the current twelve-action behavior did not establish RL deployment improvement. Both RL seeds choose exactly the same sampled action pairs as their own initialization in all four development contexts, and both underperform matched refreshed CE. This is a negative result with no engineering failure.

| Readout | Policy | New soft Dice (%) | Simulated old soft Dice (%) | Original utility |
|---|---|---:|---:|---:|
| ORIGINAL_T1_SAMPLE | REFRESH_RL | 74.172202 | 81.383098 | -0.065820113 |
| ORIGINAL_T1_SAMPLE | REFRESH_CE | 74.220562 | 81.400824 | -0.065280161 |
| ORIGINAL_T1_SAMPLE | REFRESH_INIT | 74.172202 | 81.383098 | -0.065820113 |
| EXACT_T1_EXPECTATION | REFRESH_RL | 74.086662 | 81.291493 | -0.067732576 |
| EXACT_T1_EXPECTATION | REFRESH_CE | 74.059681 | 81.302387 | -0.067896462 |
| EXACT_T1_EXPECTATION | REFRESH_INIT | 74.054055 | 81.274035 | -0.068197345 |
| FIXED_ARGMAX | REFRESH_RL | 74.103997 | 81.291709 | -0.067592478 |
| FIXED_ARGMAX | REFRESH_CE | 74.109711 | 81.321521 | -0.067222672 |
| FIXED_ARGMAX | REFRESH_INIT | 74.189137 | 81.391539 | -0.065719856 |

## Primary result and secondary readouts

Primary sampled RL-minus-CE is -0.048359 percentage points new, -0.017726 points old and -0.000539953 utility. Paired utility differences are seed601: -0.000206195, seed602: -0.000873710. Both paired RL-minus-initialization differences are exactly zero.

All eight seed/context sampled RL action pairs match initialization, which is the corresponding frozen V99 expanded CE actor. This does not mean the optimizer failed or the policy probabilities stayed unchanged: both RL fits complete1024updates, with training KL about.114, increased training expected return and about63.6%new-action probability mass. The original fixed private categorical draws still lead to the same action pairs. No extra random draws were added to seek a favorable result.

Refreshed CE improves the sampled mean over initialization, but only by about.048359pp new/.017726pp old and.000539953utility, below the original practical magnitude criteria. It also remains below the earlier expanded-initialization mean. Training-global/time both select11at100and200 and remain poor development controls. None of these findings constitutes a stable practical RL advantage.

EXACT_T1_EXPECTATION, RL-minus-REFRESH_CE: +0.026981pp new, -0.010894pp old, utility +0.000163886.

EXACT_T1_EXPECTATION, RL-minus-REFRESH_INIT: +0.032607pp new, +0.017458pp old, utility +0.000464768.

FIXED_ARGMAX, RL-minus-REFRESH_CE: -0.005714pp new, -0.029812pp old, utility -0.000369806.

FIXED_ARGMAX, RL-minus-REFRESH_INIT: -0.085140pp new, -0.099831pp old, utility -0.001872622.

The exact conditional expectation is a secondary analysis of the same frozen stream. Its modest positive utility difference does not replace the preregistered sampled endpoint. Argmax remains worse than CE and initialization. All three readouts, every seed, every context and all240sampled rows (204historical+36new) remain available; READOUT_RESULTS contains all108new mode/context/policy rows.

## Intervention and matched controls

Behavior is the equal probability ensemble of all four V99 expanded CE/RL actors, retaining the original eight training contexts, two streams and ENTRY100 checkpoints. Every first-action branch continues with this ensemble at200. Second-action returns share the actual selected first-action prefix, and the already executed selected continuation is reused once per job. The other eleven second actions run from that same prefix. No MID200 score replaces the shared MID150 term. All384training rewards are from this refreshed behavior;16current second returns are reused, no old rewards are reused in fitting.

Matched CE/RL start from the same own-seed V99 expanded CE weights and original24Dscaler, use the same384returns and1024Adamupdates each at.001withclip1. Both retain1196parameters, fixed original V94reward scale.0026099849492311478, current first-nine mean centering and clipping at3. Seven values clip. Targets, initialization and optimization costs match between CE/RL; historical methods had different collection information/cost and are labeled historical, not falsely cost-matched.

## Verification and physical accounting

All20jobs exited zero; coordinator ended and lock released. Cached-state qualification exactly reproduced104old probability vectors and8sampled pairs before collection. Native qualification passed8updates, two exact paired continuations, teacher detach and policy/student RNG isolation. Root and job ledgers agree; no failed update or recovery. All384reward formulas and behavior-state matches, all16selected-second reuse identities, every timing barrier, all240sampled/108mode rows and the decision reproduce. Initial controls match the corresponding V99CE metrics/actions in all8contexts; all204historical sampled rows remain unchanged.

New cost:56,008nativeupdates (8qualification+56,000collection),4,096actorupdates,3,712trainingimageevaluations through928paired successful query calls, zero new development queries and zero new unique images. Cached evaluation accesses576unique existing grid outcomes; weighted/readout rows are not new independent student repetitions. Cumulative campaign through V101:475,357nativeupdates excluding the common8,000source updates (483,357including),87,245actorupdates. All earlier stage costs, failures and negative conclusions remain.

## Remaining question and evidence limits

The current-policy reward refresh did not fix deployment performance. Before allocating another native collection round, the next hourly research step should separately preregister and inspect the existing training-return evidence: agreement across the two training streams, conflicting rewards for identical states, and whether state-conditioned action rankings outperform training-global/time selection when one stream is held out. This can distinguish weak or unstable reward supervision from an actor fitting problem using only already collected training rewards. Any later implementation change must receive a fresh protocol, matched controls and frozen budget before execution; do not tune against development maximizing actions or relabel this negative stage as successful.

Training roles remain16memory-fit images, support2/8from8support images and4old+4new training query images. Development uses4old+4new query images, repeatedly observed D1 and simulated retention; patient independence is not established. Exact cached readout does not restore independence. Hidden U labels, sealed test and actual later domains were not accessed. No GRPO, independent cross-domain benefit or paper-ready success is claimed. Public delivery contains source/plans/tables/aggregate diagnostics/receipts/report; weights, states, raw images and role identities remain private on NAS.
