# V108 completed: coverage helps, matched RL advantage absent

V108 completed with all 26 jobs and the coordinator exiting zero. Expanding the training table from 32 to 80 stable states improved actual sequential results relative to V106, but the preregistered RL criterion failed. RL was almost identical to analytical-target CE and slightly below 1NN. This does not qualify for fresh-stream success confirmation.

## Primary sequential outcomes

| Method | New Dice | Old Dice | Utility |
|---|---:|---:|---:|
| SAMPLE_WARM | 0.735384276 | 0.814542681 | -0.070299283 |
| SAMPLE_CE | 0.735592152 | 0.814681188 | -0.070047816 |
| SAMPLE_RL | 0.736164813 | 0.815055909 | -0.069106095 |
| SAMPLE_DISTILL | 0.736183643 | 0.815047358 | -0.069102556 |
| UNIFORM | 0.728196609 | 0.810065453 | -0.080632810 |
| GLOBAL | 0.728686742 | 0.812254683 | -0.078464928 |
| RIDGE | 0.734512831 | 0.813939322 | -0.071445044 |
| NN | 0.736031180 | 0.815277888 | -0.069023449 |

RL minus original CE: new +0.057266 percentage points, old +0.037472 points, utility +0.000941721. Both actor-seed means were positive, but stream-specific gains occurred only on stream 5; streams 3 and 4 tied. The new/old improvement did not meet the fixed practical tradeoff.

RL minus analytical-target CE: new −0.001883 percentage points, old +0.000855 points, utility −0.000003538. Seed 601 was slightly negative (−0.000007077 utility), seed 602 tied exactly. RL minus 1NN utility was −0.000082645. All eight fixed-argmax actors produced identical aggregate outcomes, so argmax is not an alternative positive result. Full context/stream/method outcomes and action distributions are retained.

## Paired effect of training coverage

| Method | New change, pp | Old change, pp | Utility change |
|---|---:|---:|---:|
| SAMPLE_WARM | 0.394275 | 0.459824 | 0.008055762 |
| SAMPLE_CE | 0.360560 | 0.407242 | 0.007140743 |
| SAMPLE_RL | 0.511741 | 0.568941 | 0.010239203 |
| SAMPLE_DISTILL | 0.519614 | 0.573100 | 0.010357854 |
| UNIFORM | 0.000000 | 0.000000 | 0.000000000 |
| GLOBAL | -0.123176 | 0.358965 | 0.002218488 |
| RIDGE | 0.299393 | 0.426649 | 0.007051485 |
| NN | 0.423296 | 0.499848 | 0.008144151 |

Every paired result uses the same context, random stream, method and entry/reference values as V106. UNIFORM was unchanged exactly. Coverage improved multiple learned controls as well as RL; it does not establish an RL-specific mechanism. Added photometric factors were already seen in development, and source cases, entry checkpoints and development streams are shared. These are paired development findings, not independent patient/domain confirmation.

## Verification and cost

All 52,105 optimizer attempt/success pairs match their child ledgers: 48,008 native and 4,097 actor updates, with zero failed calls. One ridge solve, 1,936 probes and 720 query calls / 2,880 image evaluations were recorded. All 240 reward formulas, method means, 480 decision distributions/actions and paired coverage keys were checked. All native trajectories completed before the endpoint barrier and all evaluation workers started afterward. No additional model queries were used for this report.

Cumulative through V108: 656,581 native updates excluding shared source training (664,581 including 8,000 source updates), 177,362 actor updates, and 56 linear solves (52 data, 4 synthetic). V107’s preserved metadata failure and zero-cost recovery remain in its report.

The remaining average old-reference drop for sampled RL is about 4.803 percentage points. All current 12 actions keep the unlabeled loss active, even action 0. Next, separately freeze a training-only test adding a true supervised-only action through the existing native supervised loss path. Keep old returns, conditions and states fixed and assess its action margins before any separately frozen 13-action fit/deployment. This is a new action-space hypothesis, not a revision to V108 or its thresholds.

Public delivery includes source, frozen protocol, all aggregate results and paired differences, fit summaries, action traces without private state features, accounting and this report. Images, private feature tensors, checkpoints and raw logs remain on NAS.
