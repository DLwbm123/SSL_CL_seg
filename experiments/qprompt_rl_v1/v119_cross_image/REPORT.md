# V119 cross-image target-quality diagnostic — complete

Decision: **NO_RELIABLE_CROSS_IMAGE_TARGET_QUALITY_SIGNAL**. Primary equal-context weighted net accuracy−0.002453378645 (−0.24534percentage points), weighted corrected-minus-original true-label NLL+0.041428233628 (worse). Only1/8 contexts improves both; the frozen requirement was≥6/8 plus pooled and all size/source subgroup directions. The fixed whole-mask correction remains unqualified.

Completed2026-10-09T16:57:52.233811+08:00 after66seconds of execution; exit0, once-only completion auditPASS. Run `v119_cross_20261009T085455Z`; preregistration/execution commit `84b0ea55bced3d7d239d461ad420633f3fe8dcd8` proxy-pushed/anonymous200 before launch. GPU4/NASwrite-readprobe/neutralargv and GPU process name checksPASS; no overlapping project execution. All160forwardpairs,40support exclusions,104aggregate rows and decision recomputation audited; all student/EMA/memory/optimizer tensors unchanged.

## What was evaluated

Eight existing ENTRY100 training contexts. Each current A_fit image is excluded from the support that constructs its prototype, using the fixed next two other images for n=8 and one other image for n=2. The underlying model has already trained on these images. This is prototype-held-out target-quality diagnosis, not model-held-out generalization, independent confirmation, U-label accuracy, post-training Dice or continual-learning success. Repeated image/context/pixel exposures are not independent patients. n=2 and n=8 have different support sizes as explicitly pre-registered; differences cannot be attributed solely to annotation count.

Each image uses the original EMA confidence admission, half-pixel COVERAGE and original raw class weights. Teacher/target construction uses support labels only; the target image's class labels are used only to score it (ignore255 used for validity). No Q_train/Q_dev, U image or hidden label was read. No new native or actor optimization. The query protocol and previous V118 outcomes remain unchanged.

## Primary per-context results

Positive weighted net accuracy is better; negative weighted NLL delta is better. Accuracy counts compare original versus corrected target argmax against existing known labels, not student segmentation outputs. NLL measures true-label probability in the soft pseudo-target. Values are fractions/natural-log units, not Dice. Primary aggregation weights each context equally. The count totals below pool exposures and therefore give n=8 contexts more pixels; do not substitute them for the primary mean.

| context | source_step | labeled_images | pixels | changed | repaired | harmed | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2000 | 2 | 146094 | 3755 | 608 | 3147 | -0.017002816 | 0.048496016 |
| 1 | 2000 | 2 | 146160 | 5484 | 1632 | 3852 | -0.010543562 | 0.053694968 |
| 2 | 2000 | 8 | 583423 | 22205 | 6764 | 15441 | -0.014494727 | 0.049848428 |
| 3 | 2000 | 8 | 583570 | 23207 | 6975 | 16232 | -0.015924353 | 0.053960273 |
| 4 | 8000 | 2 | 146979 | 4240 | 2323 | 1917 | 0.009901094 | 0.025958941 |
| 5 | 8000 | 2 | 146984 | 6240 | 4063 | 2177 | 0.029943402 | -0.000711757 |
| 6 | 8000 | 8 | 587956 | 21772 | 10489 | 11283 | 0.000684154 | 0.039229126 |
| 7 | 8000 | 8 | 588011 | 21701 | 9439 | 12262 | -0.002190221 | 0.060949875 |

Across all exposed selected pixels, 108604 argmax changes comprise 42293 repairs and 66311 harms (38.94% versus 61.06%). No changed-but-both-wrong pixel occurred in this diagnostic. These are repeated labeled pixel exposures, not distinct patients or U errors. Weighted repaired/harmed sums are34089.0/44618.5; weighting does not reverse the overall pooled count deficit.

## Disagreement-only results

| context | pixels | repaired | harmed | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- | --- | --- | --- |
| 0 | 3755 | 608 | 3147 | -0.603539157 | 1.721435115 |
| 1 | 5484 | 1632 | 3852 | -0.227973946 | 1.160997977 |
| 2 | 22205 | 6764 | 15441 | -0.345967558 | 1.189807756 |
| 3 | 23207 | 6975 | 16232 | -0.381179390 | 1.291640770 |
| 4 | 4240 | 2323 | 1917 | 0.257954390 | 0.676311407 |
| 5 | 6240 | 4063 | 2177 | 0.498158864 | -0.011841278 |
| 6 | 21772 | 10489 | 11283 | 0.015895818 | 0.911459409 |
| 7 | 21701 | 9439 | 12262 | -0.051933489 | 1.445214817 |

Contexts4 and6 improve weighted argmax accuracy while worsening weighted soft-target NLL; context5 alone improves both, with a small NLL benefit. Unweighted true-label NLL worsens inall8 contexts, including context5. Thus an argmax correction can coexist with worse probabilistic supervision: the log-probability costs of damaged pixels can outweigh benefits of repairs. This is observed target quality, not a measured student-gradient or final-Dice causal effect.

## All pre-registered strata

Each row's rates are equal-context means among nonempty strata; counts pool pixel exposures. These are descriptive, not posthoc eligibility gates. Full context×scope results are in RESULTS.json/csv (104rows); the table includes every original scope.

| scope | pixels | repaired | harmed | equal_context_weighted_net_accuracy | equal_context_weighted_nll_delta | positive_accuracy_contexts | improved_nll_contexts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected | 2929177 | 42293 | 66311 | -0.002453379 | 0.041428234 | 3 | 1 |
| disagreement | 108604 | 42293 | 66311 | -0.104823058 | 1.048128247 | 3 | 1 |
| memory_gate | 2891838 | 39793 | 55228 | 0.001216403 | 0.032383885 | 4 | 1 |
| no_memory_gate | 37339 | 2500 | 11083 | -0.185177711 | 0.535883656 | 0 | 0 |
| true_class_0 | 2312076 | 2010 | 46425 | -0.018812340 | 0.075416731 | 0 | 0 |
| true_class_1 | 475624 | 32328 | 19850 | 0.010633165 | 0.025464794 | 4 | 3 |
| true_class_2 | 141477 | 7955 | 36 | 0.054454442 | -0.109361443 | 8 | 8 |
| EMA_weight_class_0 | 2297748 | 19522 | 44211 | -0.011627190 | 0.050716624 | 0 | 0 |
| EMA_weight_class_1 | 482268 | 19657 | 21274 | 0.003933159 | 0.052716308 | 3 | 1 |
| EMA_weight_class_2 | 149161 | 3114 | 826 | 0.026399335 | -0.026615045 | 6 | 6 |
| confidence_0.7_0.8 | 25610 | 5862 | 7076 | 0.021344479 | -0.042067945 | 4 | 4 |
| confidence_0.8_0.9 | 39276 | 7497 | 11499 | -0.010461495 | 0.086155287 | 4 | 3 |
| confidence_0.9_1.0 | 2864291 | 28934 | 47736 | -0.001062473 | 0.038614554 | 2 | 0 |

The negative overall gate must not hide the positive class2 result: true class2 has7955repairs/36harms and improves both metrics in8/8contexts. In contrast, true background(class0) has2010repairs/46425harms, both metrics worse in8/8contexts. Thus correction introduces many false-foreground target assignments on known background while helping some foreground labels; class1 is mixed. This establishes that assignment imbalance on this labeled diagnostic, not the unobserved U data or V118 trained student predictions. True-class strata require labels and cannot be used to select U pixels at deployment.

Original-EMA confidence≥.9 yields28934repairs/47736harms, weightedNLLworse in8/8contexts. The lower confidence[.7,.8) stratum has positive pooled equal-context directions but only4/8contexts improve either separately; it is a descriptive reused slice, not a validated threshold. Original EMA class2 weight stratum improves both in6/8contexts; that also remains a diagnostic observation, not permission to retrofit the primary gate or declare an effective policy.

The memory-gate-off stratum is negative in8/8contexts on both metrics, so these data do not support simply moving all correction to that subset. Most selected pixels are memory-gate-on; its mean weightedNLLalso worsens. No claim is made that memory overwriting uniquely causes the final V118 performance loss.

## Source and label-count subgroups

| group | weighted_net_accuracy | weighted_nll_delta |
| --- | --- | --- |
| labeled_images=2 | 0.003074529 | 0.031859542 |
| labeled_images=8 | -0.007981287 | 0.050996925 |
| source_step=2000 | -0.014491365 | 0.051499921 |
| source_step=8000 | 0.009584607 | 0.031356546 |

Auxiliary8000 and n=2 subgroup mean argmax changes are positive, but soft-target NLL still worsens. No source/size subgroup improves both mean directions. Do not interpret this as a causal effect of more labels or source maturity: image sets/support sizes/model states and repeated exposures differ.

## Interpretation and limitations

The current nearest-prototype probability exchange fails its prerequisite overall target-quality check even on model-seen current labeled images. It lacks a broad empirical basis to overwrite the original target. This supports ending promotion of this exact whole-mask correction toward full RL; it does not prove prototypes contain no useful class information. The class2 improvement shows why that stronger claim would be incorrect.

All support folds contain all3classes and all3prototype center norms are above1e-8; missing-class and zero-center fallback did not occur in V119. This is newly measured for V119 and does not retrospectively establish unlogged V118 per-class norms. Full support summaries are public; image indices/support membership remain private.

Do not claim this diagnosis causally reproduces V118's training damage. It uses canonical labeled images at frozen ENTRY100 with disjoint prototype support, no training geometry/random stream, and n=2 has only1support image. It measures label quality directly but not final Dice, exact native two-image gradient magnitudes, hidden U quality or independent generalization. No OFF/fully supervised trajectory was added. No new threshold/fusion/epoch/support-set sweep or next experiment was run.

## Exact costs and delivery

40A_fit image/context reads and40 new labeled diagnostic evaluations;160 model-image forwards (40each EMA, memory, flipped memory and clean student),40 cross-image target assignments. All attempt/success pairs match. Zero optimizer/actor/solve/Q_train/Q_dev/U-image/new-annotation calls. Synthetic selfcheck0model/image/query/optimizer; the scalar audit and report also add0model/image/query/optimizer. Cumulative training remains841061native excluding8000source(849061including),181593actor,57solves; all prior failed attempts and separate512/146CPUactor-forward diagnostics preserved. This round's160segmentation forwards/40labeled evaluations are separately charged, not erased because there is no training.

Public: frozen source/protocol/selfcheck, all104anonymous aggregate rows, full positive and negative strata, support summaries, execution/costs/qualification/audit and report. Individual image metrics, original images/labels/features and weights stay on NAS. Proxy remote/anonymous verification and NAS committed-file delivery recorded in FINAL_PUBLICATION.json. No campaign success or independent confirmation.
