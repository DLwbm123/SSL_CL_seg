# V105 target-matched CE diagnostic

**Target-matched CE recovers almost all of the stable-state stream-transfer gain over the original CE objective. RL remains marginally positive in all four primary pairs, but its mean residual is only about 0.00000185 training dense reward and its argmax outcomes match distillation exactly on those pairs.**

The analytical target is the optimum of the existing KL-regularized expected-advantage objective. Plain distillation minimizes forward KL to that target; existing RL minimizes reverse KL up to a constant. This is a matched-target comparison, not identical objectives.

## All primary stream comparisons

| Representation | Test stream | Seed | RL minus DISTILL | DISTILL minus original CE | DISTILL minus1NN |
|---|---|---:|---:|---:|---:|
| ORIGINAL | stream1 | 601 | -0.000029503375 | +0.000022350256 | -0.000590723596 |
| ORIGINAL | stream1 | 602 | -0.000004930504 | +0.000094388250 | -0.000533873617 |
| ORIGINAL | stream2 | 601 | +0.000018794342 | +0.000171427495 | -0.000289560358 |
| ORIGINAL | stream2 | 602 | +0.000104063856 | +0.000246972290 | -0.000457412532 |
| FIXED_MEAN4 | stream1 | 601 | +0.000000711507 | +0.000265441779 | -0.000200780127 |
| FIXED_MEAN4 | stream1 | 602 | +0.000000874797 | +0.000243971912 | -0.000201038437 |
| FIXED_MEAN4 | stream2 | 601 | +0.000003042875 | +0.000294645452 | -0.000014065584 |
| FIXED_MEAN4 | stream2 | 602 | +0.000002768943 | +0.000290756290 | -0.000016153160 |

Mean stable-state RL-minus-DISTILL is +0.000001849530; DISTILL-minus-original-CE is +0.000273703858. Descriptively 99.3288% of the former RL-minus-CE mean is recovered by target matching. This is an algebraic comparison on the same table, not a population effect or unique causal mechanism. All four residual stream differences are positive, but their magnitude must not be hidden by a sign-only verdict.

## Condition holdout and argmax

| Representation | Mode | Seed | Mean RL minus DISTILL | Positive / zero / negative conditions | DISTILL minus CE |
|---|---|---:|---:|---|---:|
| ORIGINAL | EXACT_EXPECTATION | 601 | -0.000053923107 | 4 / 0 / 4 | +0.000241593383 |
| ORIGINAL | EXACT_EXPECTATION | 602 | -0.000031513094 | 4 / 0 / 4 | +0.000196845067 |
| ORIGINAL | ARGMAX | 601 | -0.000247373886 | 1 / 6 / 1 | +0.000400928373 |
| ORIGINAL | ARGMAX | 602 | +0.000101824757 | 3 / 5 / 0 | -0.000035114179 |
| FIXED_MEAN4 | EXACT_EXPECTATION | 601 | -0.000018143117 | 3 / 0 / 5 | +0.000242960088 |
| FIXED_MEAN4 | EXACT_EXPECTATION | 602 | +0.000006700653 | 5 / 0 / 3 | +0.000264771307 |
| FIXED_MEAN4 | ARGMAX | 601 | -0.000156980765 | 1 / 6 / 1 | +0.000214852509 |
| FIXED_MEAN4 | ARGMAX | 602 | -0.000000432308 | 2 / 5 / 1 | -0.000078666955 |

Stable-state stream argmax RL-minus-DISTILL is exactly zero for all four pairs. Stable-state condition-heldout means have mixed signs under exact expectation, and negative argmax means for both seeds. Distillation and RL both remain below1NN on stable stream means. Every condition, seed, representation and readout is retained in COMPARISONS and HELDOUT_RESULTS.

## Execution and costs

Reused all40 V104 fold-local warm actors and their frozen feature/reward normalization. Each new branch received512 full-batch fresh-Adam updates at0.001, gradient norm cap1. No extra entropy term was added. Identical normalized training inputs are grouped before the clipped advantages are averaged and the target is formed. Numerical analytical targets use float64, then probabilities cast to float32 for the actor loss.

Qualification passed the grouping, permutation, shift, forward/reverse-KL identities and one finite decreasing synthetic optimizer step. All120 old model probability/argmax readouts, covering1536 rows, replayed exactly before fitting; normalizers and reward scales also replayed exactly. All40 new actors were sealed before512 new heldout scoring rows. All512 probability/action/reward rows and every fold mean were recomputed.

Physical accounting:20,480 distillation updates plus1 synthetic update =20,481 attempted/successful optimizer calls, zero failures; exit0, process ended and lock released. CPU wall time 78.1s. No native/student update, image inference, query call or new annotation. Cumulative actor updates169,168; native remains475,357 excluding8,000 common-source updates (483,357 including). Previous54 linear solves remain separately accounted.

## Next experiment

Keep the stronger target-matched CE control. Separately preregister full-training-table stable-state policies and evaluate their sequential native execution on a fixed matrix of new random streams on the same four development conditions. Compare WARM, original CE, target-matched DISTILL, RL and simple predictors; primary stochastic and secondary deterministic readouts must remain distinct. Preserve12 actions and the existing objective/settings rather than tuning them against this diagnostic. A candidate still needs practical paired gains and a fresh fixed confirmation matrix.

These are shared-source training-table results, not Dice, sequential deployment, independent patients or new domains. The existing dense training objective differs from hinge-penalty development utility, which must remain disclosed. Repeated use of the same source/development roles does not create independent evidence.
