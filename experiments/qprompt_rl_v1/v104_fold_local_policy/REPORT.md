# V104 fold-local matched CE/RL transfer

**Stable probe features give a positive RL-minus-CE expected-reward difference in all four stream/seed pairs. This is a bounded training-table result, and the learned RL policy still trails1NN in stream transfer and both ridge/1NN on condition-heldout means.**

This training-only diagnostic completed 2026-10-08T15:18:22.195204+00:00. The result below compares exact expected **training dense reward** on heldout table rows, not Dice or sequential deployment. Every fold uses fresh, isolated fitting, and both original and stable state representations are retained.

## Primary paired results

| Representation | Test stream | Seed | RL minus matched CE | RL minus warm start | RL minus global/time | RL minus ridge | RL minus1NN |
|---|---|---:|---:|---:|---:|---:|---:|
| ORIGINAL | stream1 | 601 | -0.000007153 | +0.000085705 | +0.000005655 | +0.000028372 | -0.000620227 |
| ORIGINAL | stream1 | 602 | +0.000089458 | +0.000138156 | +0.000087078 | +0.000109794 | -0.000538804 |
| ORIGINAL | stream2 | 601 | +0.000190222 | +0.000126749 | -0.000553666 | -0.000903096 | -0.000270766 |
| ORIGINAL | stream2 | 602 | +0.000351036 | +0.000100964 | -0.000636249 | -0.000985679 | -0.000353349 |
| FIXED_MEAN4 | stream1 | 601 | +0.000266153 | +0.000289021 | +0.000653877 | +0.000227380 | -0.000200069 |
| FIXED_MEAN4 | stream1 | 602 | +0.000244847 | +0.000276848 | +0.000653782 | +0.000227285 | -0.000200164 |
| FIXED_MEAN4 | stream2 | 601 | +0.000297688 | +0.000339780 | +0.000841033 | +0.000509256 | -0.000011023 |
| FIXED_MEAN4 | stream2 | 602 | +0.000293525 | +0.000325951 | +0.000838672 | +0.000506894 | -0.000013384 |

ORIGINAL: RL beats matched CE for both seeds in both stream directions: **False**.

FIXED_MEAN4: RL beats matched CE for both seeds in both stream directions: **True**.

All four paired comparisons matter; an average does not replace a negative seed or direction. These stream folds contain the other stream of each known condition, so positive transfer here is weaker evidence than transfer to unseen conditions.

## Condition-heldout comparison

| Representation | Seed | Mean RL minus CE | Positive / zero / negative conditions | RL minus warm | RL minus global/time | RL minus ridge | RL minus1NN |
|---|---:|---:|---|---:|---:|---:|---:|
| ORIGINAL | 601 | +0.000187670 | 7 / 0 / 1 | +0.000258866 | +0.000097209 | -0.000028223 | -0.000363666 |
| ORIGINAL | 602 | +0.000165332 | 7 / 0 / 1 | +0.000299688 | +0.000194044 | +0.000068611 | -0.000266831 |
| FIXED_MEAN4 | 601 | +0.000224817 | 6 / 0 / 2 | +0.000375954 | +0.000255596 | -0.000159078 | -0.000104266 |
| FIXED_MEAN4 | 602 | +0.000271472 | 8 / 0 / 0 | +0.000381213 | +0.000193083 | -0.000221591 | -0.000166779 |

All eight individual condition folds and all new-gain/old-change components are in COMPARISONS.csv. Conditions combine existing auxiliary checkpoints, support counts and transformations from shared image/query roles; these are not independent patients or domains.

## Does stable representation improve the policies?

| Protocol | Arm | Seed | Stable minus original expected dense reward |
|---|---|---:|---:|
| STREAM_SWAP | WARM | 601 | +0.000813287 |
| STREAM_SWAP | WARM | 602 | +0.000838972 |
| STREAM_SWAP | CE | 601 | +0.000831074 |
| STREAM_SWAP | CE | 602 | +0.000971873 |
| STREAM_SWAP | RL | 601 | +0.001021460 |
| STREAM_SWAP | RL | 602 | +0.001020812 |
| CONDITION_LOCO | WARM | 601 | +0.000041299 |
| CONDITION_LOCO | WARM | 602 | -0.000082486 |
| CONDITION_LOCO | CE | 601 | +0.000121240 |
| CONDITION_LOCO | CE | 602 | -0.000107101 |
| CONDITION_LOCO | RL | 601 | +0.000158387 |
| CONDITION_LOCO | RL | 602 | -0.000000961 |

Feature stability, absolute policy quality, and RL-minus-CE gain are separate questions. Improvement in one does not establish the others.

## Fixed argmax secondary readout

| Representation | Protocol | Seed | Argmax RL minus CE mean |
|---|---|---:|---:|
| ORIGINAL | STREAM_SWAP | 601 | -0.000094539 |
| ORIGINAL | STREAM_SWAP | 602 | -0.000057427 |
| ORIGINAL | CONDITION_LOCO | 601 | +0.000153554 |
| ORIGINAL | CONDITION_LOCO | 602 | +0.000066711 |
| FIXED_MEAN4 | STREAM_SWAP | 601 | +0.000031887 |
| FIXED_MEAN4 | STREAM_SWAP | 602 | +0.000029750 |
| FIXED_MEAN4 | CONDITION_LOCO | 601 | +0.000057872 |
| FIXED_MEAN4 | CONDITION_LOCO | 602 | -0.000079099 |

Argmax was frozen as secondary, with lowest-action-ID ties. It cannot replace the exact-expectation primary. No sampled readout, temperature search, posthoc checkpoint selection or additional seed was introduced.

## Fitting and objective diagnostics

The native24->32tanh->12architecture has1,196parameters. Fresh seed601/602hidden weights and a zero final head give a uniform initial policy. Each representation/fold/seed uses512CEwarmup updates, then two copied warm actors receive512matched CE or RL continuation updates using fresh Adam(.001) and gradient norm cap1. Warmup optimizer state is discarded for both branches. Each endpoint therefore reflects1,024updates along its path, with the512warmup shared and charged once.

Feature mean/std and the first-nine-action reward RMS are fitted using training rows only; numerical reward floor is fixed1e-4. Row-centered all12advantages are clipped[-3,3]. The existing objectives are reused: CE soft-target temperature.25 and entropy bonus.01; RL expected advantage, KL coefficient.25 to its own fold-local warm actor, and entropy bonus.01. Existing full-table actors, feature normalizers and data-derived reward floors are not used. Neural arithmetic is float32, consistent with the existing architecture; readout probabilities are normalized in float64 for exact sums.

| Representation | Arm | Mean training RL-objective gap | Mean training expected dense reward | Mean KL to warm |
|---|---|---:|---:|---:|
| ORIGINAL | WARM | 0.094350014 | 0.005973511 | 0.000000000 |
| ORIGINAL | CE | 0.048126961 | 0.006113913 | 0.040946526 |
| ORIGINAL | RL | 0.001927937 | 0.006312101 | 0.143624109 |
| FIXED_MEAN4 | WARM | 0.102807052 | 0.005945923 | 0.000000000 |
| FIXED_MEAN4 | CE | 0.077860270 | 0.006022586 | 0.025384594 |
| FIXED_MEAN4 | RL | 0.001344062 | 0.006317467 | 0.161644299 |

Training objective gaps compare with the per-identical-feature optimum after averaging already clipped advantages; every gap satisfies the scaled-KL identity. This diagnoses fitting under the frozen objective and includes shared-function constraints. CE has a different objective, so a larger RL-objective gap does not imply CE optimization failure. The larger table contains all120model diagnostics, scales, clipping counts and three loss checkpoints per fit.

## Costs and verification

The preregistered61442actor optimizer steps completed:20480warmup +20480CE +20480RL +2synthetic qualification. All61442attempts have matching successes, with zero failure. The immutable raw ledger stays onNAS; public PHYSICAL_ACCOUNTING.json contains category and per-fit counts, including the two synthetic steps. Forty fold/seed/representation pairs produced120private actor files, all sealed before any heldout scoring. No pretrained full-table actor entered fitting. No new ridge fit was performed;640exactly matching V103control rows are reused and disclosed.

All1536heldout rows, their probabilities/argmax and reward components were recomputed against the frozen V101training table. All fold summaries and control reuse were checked. Synthetic CE/RL objectives decreased after their qualification updates; initialization identity, global RNG preservation and constant-feature handling passed. Live process identity and neutral command line were verified, final OS exit0 was captured, the process ended and lock released. The error log was empty.

CPU wall time from launch to final receipt was230.8seconds. Cumulative campaign actor updates are148,687(87,245prior +61,442this stage); native updates remain475,357excluding8,000common-source updates, or483,357including. Prior50data+4synthetic linear solves remain separate. No native/student update, image inference, reward query or new annotation case occurred in this stage.

This evaluates single-decision action values on a frozen behavior-state table, whose first-action returns include a frozen continuation. It does not execute the learned policy through a new student trajectory. No Qdev/grid outcome or real later domain was read. Existing supervision remains16memory-fit, support2/8from8support images,4old+4new Q_train and historical4old+4new Q_dev; two-support operation does not erase meta-training label costs. Hidden U labels remain unused.

## Interpretation and next control

Stable-state RL gains over matched CE range+.000244847 to+.000297688 across the four primary stream/seed pairs. Under condition holdout, both seed means are positive, but seed601still has two negative conditions. Stable features improve RL strongly on stream transfer (+.001021approximately perseed), while condition-heldout improvement is+.000158387 forseed601 and-.000000961 forseed602. Generalization beyond known conditions is therefore unresolved.

A descriptive decomposition of the already frozen readouts separates the expected increment into the change in modal-action reward and the change in reward from mixing over actions. This is algebra on the preregistered outputs, not a new temperature/readout selection, and is marked exploratory:

| Representation | Protocol | Mean expected RL-minus-CE | Modal-action component | Relative mixture component |
|---|---|---:|---:|---:|
| ORIGINAL | STREAM_SWAP | +0.000155891 | -0.000075983 | +0.000231874 |
| ORIGINAL | CONDITION_LOCO | +0.000176501 | +0.000110133 | +0.000066369 |
| FIXED_MEAN4 | STREAM_SWAP | +0.000275553 | +0.000030818 | +0.000244735 |
| FIXED_MEAN4 | CONDITION_LOCO | +0.000248144 | -0.000010614 | +0.000258758 |

Most of the stable-state stream increment disappears under argmax, and the stable-state condition-heldout mean argmax increment is slightly negative. The result is mainly a probability-distribution improvement relative to this CE target, with limited evidence for better modal action selection. A mixture component is not by itself a causal entropy effect: nonmodal action probabilities and their values both matter.

A useful next isolated control is to train CE against the analytical optimum of the actual RL training objective, using exactly the same fold-local warm policy and512update budget. The target is proportional to exp((mean_clipped_advantage+.25log_warm)/.26), grouped only for identical train features. Plain cross-entropy to this target minimizes forward KL, while the existing RL objective minimizes reverse KL to the same target up to a constant. No extra entropy term should be added to the distillation objective, which would change its optimum. This tests whether the incremental gain comes from the changed target distribution or the optimization objective under the shared actor. The existing CE targets are not matched to that analytical target. This control needs its own preregistration, fixed cost and synthetic checks; it has not run. No new native rollout, query, feature extraction or independent-source evaluation is implied.
