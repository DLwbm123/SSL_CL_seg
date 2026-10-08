# V102 training reward reliability and state prediction

**Training rankings transfer across the two streams, but stable prediction from the current state representation is not established. The existing RL actors already nearly optimize their frozen training objective.** This CPU diagnostic completed 2026-10-08T14:00:35.739578+00:00; it adds no native update, actor optimizer update, image query or deployment result.

This narrows the next question toward transferable state-to-action mapping and task coverage. It does not prove fundamental state insufficiency or identify a unique cause of the development failures.

## Reward rankings are not simply unrepeatable

| Dataset | Decision step | Mean Spearman | Pairwise ordering agreement | Same top action |
|---|---:|---:|---:|---:|
| V101 | 100 | 0.908217 | 90.720% | 87.5% |
| V101 | 200 | 0.852273 | 88.636% | 87.5% |
| V99 | 100 | 0.888986 | 89.205% | 62.5% |
| V99 | 200 | 0.847028 | 89.015% | 87.5% |

V101 top actions agree in14of16context/timepairs. Mean top-versus-runner-up raw dense reward gaps are only about.00036-.00040. Choosing each context/time action using one stream and scoring it on the other beats source-stream-trained global/time choices in both directions:

| Table | Source stream | Heldout dense reward difference | New-gain component difference | Old-change component difference |
|---|---:|---:|---:|---:|
| V101 | 1 | +0.000852056 | +0.000359885 | +0.000492171 |
| V101 | 2 | +0.000853946 | +0.000320577 | +0.000533368 |
| V99 | 1 | +0.000611552 | +0.000248649 | +0.000362902 |
| V99 | 2 | +0.000805478 | +0.000311996 | +0.000493482 |

Global and time baselines coincide here. This context-matched transfer uses the other stream's action values for that known condition; it is a diagnostic information advantage, not a deployable state predictor. Scores are training dense reward, weighted new-task gain and old-query change, not development Dice or the hinge-based development utility.

Both tables contain32unique24Dstate vectors. No cross-stream pair has an identical state vector or identical complete snapshot. All eight step100pairs have identical student weights but different stream/provider/RNG inputs; step200weights differ after adaptation. Therefore these observations measure cross-stream transfer. They do not identify same-state reward noise, even at the first decision point. The earlier possibility of repeated identical states is not realized in these tables.

## Current state prediction has limited, inconsistent evidence

All predictors are fitted independently inside each fold. Ridge uses only train-fold feature means/scales and row-centered raw rewards, a fixed lambda1, and an unpenalized intercept.1NN uses the same train-only scaling. No pretrained actor or previous actor scaler enters these folds; existing CE/RL are not misrepresented as heldout models. There is no hyperparameter search.

| Table | Method | Test stream1 minus global/time | Test stream2 minus global/time | Positive in both directions? |
|---|---|---:|---:|---|
| V101 | STATE_RIDGE | -0.000022716 | +0.000349430 | False |
| V101 | STATE_1NN | +0.000625882 | -0.000282900 | False |
| V99 | STATE_RIDGE | +0.000176621 | +0.000373290 | True |
| V99 | STATE_1NN | +0.000605992 | -0.000141195 | False |

Primary V101 ridge and1NN each improve only one stream direction. The other direction is negative; the small average improvements must not hide that instability. Secondary V99 ridge is positive in both directions, so these diagnostics do not establish that current states contain no useful signal.

| Table | Condition-heldout method | Mean dense reward difference vs global/time | Positive / zero / negative conditions |
|---|---|---:|---|
| V101 | STATE_RIDGE | +0.000125433 | 2 / 3 / 3 |
| V101 | STATE_1NN | +0.000460875 | 6 / 0 / 2 |
| V99 | STATE_RIDGE | +0.000060436 | 3 / 3 / 2 |
| V99 | STATE_1NN | +0.000486963 | 6 / 0 / 2 |

All640individual heldout evaluations and every fold are retained. The eight conditions are combinations of two auxiliary checkpoints, two support counts and two transformations sharing source/query roles. Leaving one condition out is not independent patient or source validation. Such a source-independent test is marked NA; no new independence is manufactured by regrouping. Two fixed predictors and one ridge setting also cannot exhaust all functions of the available state.

## The RL training objective is already nearly solved

For the exact V101 clipped advantages and own-seed V99CE reference, the per-state optimum is softmax((mean_clipped_advantage+.25log_reference)/.26). Identical state vectors would be grouped and their already clipped advantages averaged before solving; all32states happen to be unique. Forward log probabilities are numerically normalized in float64 for analytical arithmetic. No oracle is deployed or scored on development data.

| Seed | Initial objective gap | CE objective gap | RL objective gap | Initial-to-optimum gap closed by RL |
|---|---:|---:|---:|---:|
| 601 | 0.049033650 | 0.047662861 | 0.000205951 | 99.5800% |
| 602 | 0.048906137 | 0.047416679 | 0.000199886 | 99.5913% |

Every group/seed gap satisfies Jstar-Jactor=.26KL(actor||star) within numerical tolerance. Both RL actors close about99.6%of the available improvement in this training objective. This strongly weakens an explanation based primarily on failure to optimize that objective. CE optimizes a different loss, so its larger gap under the RL objective is not evidence that CE training failed. The unrestricted-versus-shared-actor gap includes representation/capacity constraints as well as optimization; it is not a pure optimizer diagnostic. Near-optimal fit to training states does not establish generalization.

## Costs, verification and next direction

The run consumed20closed-form data ridge solves and2synthetic qualification solves, each with paired attempt/success records; no failed solve. The20nearest-neighbor reference sets require no optimization. Six frozen actor evaluations support the analytical comparison. No native/actor optimizer update, training/development image query or new unique image. Cumulative native/actor campaign totals remain475,357/87,245excluding8,000common-source updates (483,357nativeincluding). This is a completed diagnostic, not another performance replication.

Dataset dimensions, action keys, finite values and dense reward decomposition passed. All640evaluation keys are unique, fold aggregates reproduce, analytical identities hold, the final receipt exists, process ended and lock released. The error log is empty; an OS exit code was not separately captured and is not claimed. The initial structure-only inspection was disclosed before preregistration. Code, plans, every result and solve accounting are published; private checkpoint/state tensors remain onNAS.

A justified next small intervention is to test whether the current single-batch state is unstable: compare the original features with a fixed, label-role-safe state probe averaged over a few predetermined batches, using the same frozen training checkpoints and these same training rewards. First reproduce original states exactly, ensure inference does not modify student/RNG state, then use the same train-only folds and predictors. This needs its own preregistration and bounded inference budget; it does not justify another native reward collection or a GRPO/PPO hyperparameter sweep. If stable state features still do not transfer, reconsider task coverage or action-relevant information before more optimization.

All evidence remains shared-source training diagnostics and repeatedly observed D1 history, with16memory-fit images, support2/8from8support images and4old+4new training-query images. No development grid was read by this diagnostic. Hidden U labels, sealed test and real later domains remain untouched. Method key isolation, A-only adaptation and history-free parameter memory are unchanged.
