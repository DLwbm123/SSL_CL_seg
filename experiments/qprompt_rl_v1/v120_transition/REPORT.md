# V120 observable-transition target quality — complete

Decision: **OBSERVABLE_TRANSITION_QUALITY_SIGNAL**. Qualifying transitions: **1->0, 2->1**. This is development-only target quality; no training benefit, RL gain, independent confirmation or campaign success is established.

Completed 2026-10-09T17:21:30.346763+08:00, elapsed 99.4seconds, root EXIT0 and once-only auditPASS. Run `v120_transition_20261009T091857Z`; pre-execution commit `231ab451ee2469ff01f83c5758267f7839500360` was proxy-pushed with remote/anonymous200 verified before launch. GPU4 admission, NAS mount/write/read, no overlapping EXEC_RUN, neutral process/GPU argv checksPASS. Exactly160forward attempt/success pairs,40prototype-held-out evaluations,296aggregate rows and all transition/truth partitions audited. Models and optimizer unchanged. Selected-all rows exactly reproduce V119, confirming measurement consistency, not independent replication.

## Question and evidence boundary

V119's overall correction failed and true-class2 improvements could not be used as an unlabeled selector. This round measures the missing joint relation between original target argmax, prototype-corrected argmax and known truth. Original-to-corrected transitions are observable; truth is used only for scoring/audit. All six off-diagonal actions are evaluated without choosing a favorable condition, seed, threshold or checkpoint. Models already trained on these40repeated A_fit image/context exposures; only prototype construction excludes each target image. These are not40independent patients or an independent holdout. No U labels, Q_train or Q_dev are read.

Every candidate action means exchange probabilities only on that transition and leave all other selected pixels unchanged. Whole-selected effect divides by all original selected weight in its context, so tiny subsets cannot inflate the mean; empty actions contribute zero. Per-transition conditional rates are separately available in RESULTS.json/csv and must not be confused with whole-selected effects. Positive accuracy and negative true-label NLL delta are better. NLL is diagnostic target quality, not the actual native KL(target||student) loss or a gradient magnitude.

## All six pre-registered candidate actions

Sign gate: both mean directions, jointly positive in≥6/8contexts, and both directions in every n=2/n=8/source2000/source8000 subgroup. This is a descriptive development qualification across six tested candidates, not a significance test or a replacement for historical practical/absolute/matched-CE/fresh-stream requirements. No selected candidate is deployed or trained in this round.

| transition | pixels | repaired | harmed | equal_context_weighted_net_accuracy | equal_context_weighted_nll_delta | joint_positive_contexts | nonempty_contexts | qualified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0->1 | 76946 | 30521 | 46425 | -0.000560181 | 0.024305455 | 0 | 8 | False |
| 0->2 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0 | 0 | False |
| 1->0 | 2015 | 2010 | 5 | 0.000418375 | -0.000384184 | 6 | 6 | True |
| 1->2 | 27800 | 7955 | 19845 | -0.003927429 | 0.018997224 | 1 | 8 | False |
| 2->0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0 | 0 | False |
| 2->1 | 1843 | 1807 | 36 | 0.001615857 | -0.001490262 | 8 | 8 | True |

## Complete joint assignment table

Counts pool all repeated selected pixel exposures and therefore weight n=8 contexts more heavily; primary gate means weight contexts equally. Diagonal entries retain original assignment. For off-diagonal a→b, truth=a is a harmed correct target, truth=b is repaired, and the third class is changed-but-still-wrong. This table directly identifies false assignments entering each proposed class; it does not measure full-image student Dice or U-data precision.

| transition | pixels | truth_0 | truth_1 | truth_2 | repaired | harmed | both_wrong_changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0->0 | 2226324 | 2212638 | 13686 | 0 | 0 | 0 | 0 |
| 0->1 | 76946 | 46425 | 30521 | 0 | 30521 | 46425 | 0 |
| 0->2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1->0 | 2015 | 2010 | 5 | 0 | 2010 | 5 | 0 |
| 1->1 | 444786 | 51003 | 388533 | 5250 | 0 | 0 | 0 |
| 1->2 | 27800 | 0 | 19845 | 7955 | 7955 | 19845 | 0 |
| 2->0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2->1 | 1843 | 0 | 1807 | 36 | 1807 | 36 | 0 |
| 2->2 | 149463 | 0 | 21227 | 128236 | 0 | 0 | 0 |

## Every candidate's context and subgroup results

All effects below use the whole-selected denominator; zero means no net effect, not missing data. All positive and negative results are retained.


### 0->1

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 2310 | -0.009906819 | 0.025761600 |
| 1 | 2000 | 2 | 3294 | -0.006844066 | 0.026976485 |
| 2 | 2000 | 8 | 15773 | -0.005954237 | 0.030628421 |
| 3 | 2000 | 8 | 17570 | -0.005725074 | 0.032311025 |
| 4 | 8000 | 2 | 2888 | 0.007081195 | 0.012957049 |
| 5 | 8000 | 2 | 3974 | 0.013116707 | 0.009229244 |
| 6 | 8000 | 8 | 14606 | 0.000508075 | 0.024368971 |
| 7 | 8000 | 8 | 16531 | 0.003242770 | 0.032210842 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000861754 | 0.018731095 |
| labeled_images=8 | -0.001982116 | 0.029879815 |
| source_step=2000 | -0.007107549 | 0.028919383 |
| source_step=8000 | 0.005987187 | 0.019691527 |

### 0->2

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 3 | 2000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 4 | 8000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 5 | 8000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 6 | 8000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 7 | 8000 | 8 | 0 | 0.000000000 | 0.000000000 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000000000 | 0.000000000 |
| labeled_images=8 | 0.000000000 | 0.000000000 |
| source_step=2000 | 0.000000000 | 0.000000000 |
| source_step=8000 | 0.000000000 | 0.000000000 |

### 1->0

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | 227 | 0.000295533 | -0.000253503 |
| 3 | 2000 | 8 | 19 | 0.000025918 | -0.000009883 |
| 4 | 8000 | 2 | 157 | 0.000822907 | -0.000628442 |
| 5 | 8000 | 2 | 9 | 0.000047429 | -0.000024225 |
| 6 | 8000 | 8 | 1518 | 0.002040366 | -0.002082747 |
| 7 | 8000 | 8 | 85 | 0.000114848 | -0.000074670 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000217584 | -0.000163167 |
| labeled_images=8 | 0.000619166 | -0.000605201 |
| source_step=2000 | 0.000080363 | -0.000065847 |
| source_step=8000 | 0.000756388 | -0.000702521 |

### 1->2

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 1333 | -0.008283967 | 0.023437802 |
| 1 | 2000 | 2 | 1851 | -0.007282721 | 0.030118269 |
| 2 | 2000 | 8 | 6201 | -0.008846918 | 0.019478798 |
| 3 | 2000 | 8 | 5439 | -0.010653521 | 0.021979764 |
| 4 | 8000 | 2 | 1075 | 0.000739044 | 0.014580379 |
| 5 | 8000 | 2 | 1787 | 0.011867747 | -0.004905500 |
| 6 | 8000 | 8 | 5545 | -0.002141175 | 0.017217094 |
| 7 | 8000 | 8 | 4569 | -0.006817924 | 0.030071188 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | -0.000739974 | 0.015807737 |
| labeled_images=8 | -0.007114884 | 0.022186711 |
| source_step=2000 | -0.008766782 | 0.023753658 |
| source_step=8000 | 0.000911923 | 0.014240790 |

### 2->0

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 1 | 2000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 2 | 2000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 3 | 2000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 4 | 8000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 5 | 8000 | 2 | 0 | 0.000000000 | 0.000000000 |
| 6 | 8000 | 8 | 0 | 0.000000000 | 0.000000000 |
| 7 | 8000 | 8 | 0 | 0.000000000 | 0.000000000 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.000000000 | 0.000000000 |
| labeled_images=8 | 0.000000000 | 0.000000000 |
| source_step=2000 | 0.000000000 | 0.000000000 |
| source_step=8000 | 0.000000000 | 0.000000000 |

### 2->1

| context | source_step | labeled_images | pixels | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 112 | 0.001187970 | -0.000703386 |
| 1 | 2000 | 2 | 339 | 0.003583225 | -0.003399786 |
| 2 | 2000 | 8 | 4 | 0.000010895 | -0.000005288 |
| 3 | 2000 | 8 | 179 | 0.000428323 | -0.000320633 |
| 4 | 8000 | 2 | 120 | 0.001257947 | -0.000950046 |
| 5 | 8000 | 2 | 470 | 0.004911519 | -0.005011277 |
| 6 | 8000 | 8 | 103 | 0.000276888 | -0.000274193 |
| 7 | 8000 | 8 | 516 | 0.001270085 | -0.001257486 |

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.002735165 | -0.002516124 |
| labeled_images=8 | 0.000496548 | -0.000464400 |
| source_step=2000 | 0.001302603 | -0.001107273 |
| source_step=8000 | 0.001929110 | -0.001873250 |

## Interpretation

The original whole-mask method remains rejected by V118 continuous-training and V119 target-quality outcomes. An observable transition that passes this round would be a concrete candidate for a separately frozen training comparison, not proof that local corrections improve future optimization. In particular, V119's sole joint-positive context5 had V118's largest new-task training loss. Do not cherry-pick that context, use true-class labels for U selection, retrofit a confidence threshold, or infer an RL benefit from deterministic selection.

The same canonical states, support ordering, seed119 and labels are deliberately reused to add unlogged joint statistics. Agreement with V119 is internal consistency, not new independent supporting evidence. Support size differs(n=2 has1other image,n=8 has2); context repetitions and source/model differences preclude a causal annotation-count conclusion. Pixel exposures are correlated. Native training uses augmented U images with two-image batches and model updates, which are absent here. The visible transition definition is original blended-target argmax→actual corrected argmax, not EMA class→true class. Probability multisets/entropy preservation does not establish target correctness.


## What the new joint table resolves

All46425 harmed known-background assignments enter class1 through0→1; none enter class2. Class2 gains7955 true-class2 repairs through1→2, but that same transition creates19845 false-class2 assignments from true class1. Thus the previous true-class2 improvement was not evidence that adding class2 predictions is reliable. Conversely2→1 repairs1807 and harms36;1→0 repairs2010 and harms5. These findings concern current labeled, selected target assignments, not U pixels or trained student predictions.

The two qualifying actions are sparse:1→0 acts on2015 repeated pixels and2→1 on1843, together3858/2929177 selected exposures(about0.132%). Their equal-context weighted whole-selected accuracy effects are respectively+0.04184 and+0.16159percentage points; these are target accuracy, not Dice.1→0 passes in six nonempty contexts and is absent in contexts0/1;2→1 passes in8/8 but has only4 changed exposures in context2. Consistency does not make the correlated small subsets independent or establish a practical training margin.

## Additional evidence required before promotion

The latest research review adds an important untested alternative: the blended EMA/memory target can disagree with EMA, so prototype correction may merely restore EMA's existing decision. V120 does not record the required EMA×original×proposal joint comparison and cannot establish incremental prototype information. Do not infer that either qualifying transition beats EMA or memory.

The next separately pre-registered mechanism comparison should include the original target, the prototype action and direct EMA-class probability exchange on the exact same candidate region, using the same probability-exchange operation. If EMA agrees with the old target there, its exchange is identity; record actual changed counts rather than pretending every arm has equal dose. This same-region comparison asks whether the prototype provides value beyond an available prediction, not whether a more complex rule beats full harmful rewriting.

A complementary location-shuffled control should preserve observable original-class strata, destination-class counts and the number of changed pixels, while relocating suggestions and no-change tokens across the broader selected set using a fixed label-free permutation. Preserve original EMA weight strata where feasible to match supervised mass. Merely shuffling identical destination labels within an already fixed single transition is vacuous and cannot test local information. Frozen identity/no-move cases must be disclosed; no outcome-selected permutation or repeated shuffle search. This resolves the unmatched99.565% versus7.971% intervention problem of V118's ROTATED control without claiming that this control has already run.

Existing eight seen training conditions can only support a development mechanism comparison. Before claims about unseen-image reliability, separately define a legal training-only model-level holdout whose target images are absent from all model fitting, not merely absent from their prototype support. Do not relabel V119/V120 as this holdout or open sealed Q_dev/test/hidden U labels. Do not pick thresholds on these eight conditions and call the same conditions validation. A subsequent short paired training comparison must retain same entry, stream and budget, check new-task gain and old-task harm, and include a mass-matched control if U strength or selected mass changes.

Retain the history-free backbone and trained-parameter memory. Prototypes remain transient current-domain evidence, not a new historical bank. The overall probability-exchange action remains rejected; no GRPO expansion or other full RL run follows from this diagnostic alone. Positive target-quality subsets are candidates, not proof of extra information or training value. These requirements are integrated into campaign handoff; none of these additional controls or holdouts was executed in V120.

## Costs and public delivery

This round adds160segmentation image-forwards(40each EMA/memory/flipped-memory/student),40current-labeled image/context reads/evaluations and40prototype assignments. Combined V119+V120:320forwards/80evaluations; the new inference is charged again and V119 is retained. Zero native/actor optimizer updates, solves, Q_train/Q_dev queries, U-image reads, hidden labels or new annotations. Cumulative training unchanged841061native excluding8000source(849061including),181593actor,57solves; all historical failures and separate512/146CPUactor-forward diagnostics preserved. Synthetic selfcheck and scalar completion audit add zero model/image/optimization calls.

Public: frozen source/protocol/selfcheck, all296anonymous aggregate rows, all6candidate/context/subgroup decisions, complete pooled joint table, support/consistency/qualification/cost/execution/audit/report. Individual image metrics, support IDs, labels/features/data/weights/raw logs remain private on NAS. Delivery uses committed files, proxy GitHub push, remote branch/anonymous report check and NAS FINAL_PUBLICATION.json. No next training experiment was launched during this closeout.

Operational note: after scalar files had transferred successfully, the local synchronization helper raised a KeyError while printing a summary due to a tuple-key typo. No experiment, inference, transfer or create-only completion audit was rerun. Report generation used the already transferred audited files. This post-processing print failure is retained and adds zero model/image/training cost.
