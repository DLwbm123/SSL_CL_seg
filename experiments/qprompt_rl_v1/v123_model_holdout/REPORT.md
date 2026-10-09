# V123 — model-held-out EMA conflict rule: complete

Primary status: **STOP_NO_MODEL_HELD_OUT_SIGNAL**. Stage A quality gate **FAILED**; conditional Stage B five-arm training **NOT RUN**, as preregistered. This is a completed negative prerequisite experiment, not an engineering failure, training-benefit result, independent confirmation or RL success.

Finished 2026-10-09T18:27:50.457729+08:00, elapsed 435.5 seconds. Root and all65 executed jobs exited0; the once-only completion audit PASS. Execution/preregistration commit`212c734d27ffda8e0b1bc75ded5efc15600fcedf` was proxy-pushed and anonymously accessible before launch. Run`v123_holdout_20261009T101910Z`. GPU4–7, NAS mount/write/read probe, free-memory admission, neutral coordinator/child argv and GPU process identity verified. No restart, retry, tuning, extra seed, annotation or hidden-label access.

## Primary result

Equal32 condition/stream mean weighted net target accuracy: **-0.001436382143**, or **-0.143638 percentage points**. Weighted true-label NLL delta corrected−base: **+0.001803070612** (worse). Only **4/16** context means were jointly positive; the frozen requirement was≥12/16 plus both pooled signs and every subgroup's two signs. At individual stream/condition level9/32 were jointly positive, descriptive only and not a substitute gate.

Pooled correlated pixel exposures:9,367,918 selected;63,039 changed;24,341 repaired;38,674 harmed;24 changed while both targets remained wrong. Weighted repair mass19,252.0 versus harm27,960.5. These counts are not independent samples or the equal-condition primary mean. The unchanged majority dilutes whole-selected accuracy differences; no inference about deployment pixel reliability, hard Dice, clinical endpoints or statistical independence follows.

## Every subgroup

| group | weighted target accuracy delta (pp) | weighted NLL delta |
| --- | --- | --- |
| fold=0 | -0.35042101 | +0.0041082988 |
| fold=1 | +0.06314458 | -0.0005021576 |
| stream=3 | -0.15133734 | +0.0018698980 |
| stream=4 | -0.13593909 | +0.0017362432 |
| source_step=2000 | -0.20755503 | +0.0024716601 |
| source_step=8000 | -0.07972140 | +0.0011344811 |
| labeled_images=2 | -0.02873311 | +0.0006049955 |
| labeled_images=4 | -0.25854332 | +0.0030011458 |
| condition=brightness | -0.27365428 | +0.0029604586 |
| condition=contrast | -0.01362215 | +0.0006456826 |

Both streams, both auxiliary-source groups, both annotation budgets and both photometric-condition groups have negative mean accuracy and worse mean NLL. Fold1 is positive on average, while fold0 is negative; this does not license selecting fold1 or excluding unfavorable images. No claim that all individual conditions failed.

## All16 context means (two streams equally averaged)

| context | fold | source step | fit images | condition | accuracy delta (pp) | NLL delta |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 2000 | 2 | brightness | -0.54918953 | +0.0053122856 |
| 1 | 0 | 2000 | 2 | contrast | +0.08563149 | +0.0000740619 |
| 2 | 0 | 2000 | 4 | brightness | -0.55347574 | +0.0053536106 |
| 3 | 0 | 2000 | 4 | contrast | -0.69877735 | +0.0089451810 |
| 4 | 0 | 8000 | 2 | brightness | -0.45577186 | +0.0051498115 |
| 5 | 0 | 8000 | 2 | contrast | +0.00575786 | +0.0004443742 |
| 6 | 0 | 8000 | 4 | brightness | -0.44243463 | +0.0048508267 |
| 7 | 0 | 8000 | 4 | contrast | -0.19510829 | +0.0027362389 |
| 8 | 1 | 2000 | 2 | brightness | +0.15079050 | -0.0011098250 |
| 9 | 1 | 2000 | 2 | contrast | +0.24034003 | -0.0022525480 |
| 10 | 1 | 2000 | 4 | brightness | -0.30123283 | +0.0032814397 |
| 11 | 1 | 2000 | 4 | contrast | -0.03452685 | +0.0001690750 |
| 12 | 1 | 8000 | 2 | brightness | -0.00946908 | +0.0003588016 |
| 13 | 1 | 8000 | 2 | contrast | +0.30204570 | -0.0031369981 |
| 14 | 1 | 8000 | 4 | brightness | -0.02845110 | +0.0004867183 |
| 15 | 1 | 8000 | 4 | contrast | +0.18566024 | -0.0018139239 |

## All32 condition/stream results

| context | stream | selected | changed | repaired | harmed | both wrong changed | accuracy delta (pp) | NLL delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 291383 | 2136 | 412 | 1719 | 5 | -0.51571670 | +0.0049910356 |
| 0 | 4 | 291249 | 2235 | 372 | 1853 | 10 | -0.58266236 | +0.0056335356 |
| 1 | 3 | 291020 | 2758 | 1301 | 1457 | 0 | +0.10707351 | -0.0002596837 |
| 1 | 4 | 290846 | 3080 | 1387 | 1693 | 0 | +0.06418947 | +0.0004078074 |
| 2 | 3 | 291232 | 1903 | 216 | 1678 | 9 | -0.60269733 | +0.0058452326 |
| 2 | 4 | 291054 | 1777 | 299 | 1478 | 0 | -0.50425414 | +0.0048619885 |
| 3 | 3 | 290996 | 4772 | 1381 | 3391 | 0 | -0.69411596 | +0.0088983387 |
| 3 | 4 | 290967 | 4860 | 1401 | 3459 | 0 | -0.70343874 | +0.0089920233 |
| 4 | 3 | 293576 | 3226 | 1073 | 2153 | 0 | -0.46823165 | +0.0052483614 |
| 4 | 4 | 293599 | 3292 | 1159 | 2133 | 0 | -0.44331206 | +0.0050512615 |
| 5 | 3 | 293833 | 2210 | 1079 | 1131 | 0 | -0.00393287 | +0.0005220052 |
| 5 | 4 | 293771 | 2500 | 1250 | 1250 | 0 | +0.01544859 | +0.0003667432 |
| 6 | 3 | 293674 | 2353 | 607 | 1746 | 0 | -0.51621488 | +0.0056855728 |
| 6 | 4 | 293665 | 1959 | 601 | 1358 | 0 | -0.36865439 | +0.0040160807 |
| 7 | 3 | 293765 | 2713 | 1137 | 1576 | 0 | -0.14644183 | +0.0021246700 |
| 7 | 4 | 293711 | 3195 | 1261 | 1934 | 0 | -0.24377476 | +0.0033478078 |
| 8 | 3 | 292268 | 625 | 448 | 177 | 0 | +0.13717233 | -0.0009763362 |
| 8 | 4 | 292155 | 709 | 520 | 189 | 0 | +0.16440868 | -0.0012433138 |
| 9 | 3 | 292325 | 591 | 491 | 100 | 0 | +0.20724052 | -0.0019281466 |
| 9 | 4 | 292337 | 690 | 607 | 83 | 0 | +0.27343954 | -0.0025769494 |
| 10 | 3 | 292206 | 1734 | 325 | 1409 | 0 | -0.30023401 | +0.0032748672 |
| 10 | 4 | 292212 | 1757 | 341 | 1416 | 0 | -0.30223165 | +0.0032880122 |
| 11 | 3 | 292404 | 343 | 114 | 229 | 0 | -0.04474322 | +0.0002699718 |
| 11 | 4 | 292381 | 384 | 149 | 235 | 0 | -0.02431047 | +0.0000681782 |
| 12 | 3 | 293865 | 1929 | 925 | 1004 | 0 | -0.02327299 | +0.0005132717 |
| 12 | 4 | 293838 | 1986 | 983 | 1003 | 0 | +0.00433483 | +0.0002043314 |
| 13 | 3 | 293837 | 1308 | 982 | 326 | 0 | +0.31660745 | -0.0032931986 |
| 13 | 4 | 293952 | 1140 | 864 | 276 | 0 | +0.28748396 | -0.0029807976 |
| 14 | 3 | 294033 | 1657 | 752 | 905 | 0 | -0.03830422 | +0.0005800876 |
| 14 | 4 | 294030 | 1638 | 804 | 834 | 0 | -0.01859799 | +0.0003933490 |
| 15 | 3 | 293873 | 700 | 486 | 214 | 0 | +0.16441445 | -0.0015776816 |
| 15 | 4 | 293861 | 879 | 614 | 265 | 0 | +0.20690604 | -0.0020501662 |

## What was actually tested

Two deterministic image folds split the canonical eight A_fit images4/4. Current models fit either two or four images within the fit half; the opposite four are absent from source/current fitting and U. The existing auxiliary models use disjoint M_fit16. No prior A_fit-trained ENTRY was reused. Each of16 conditions × two training streams was rebuilt for100 native action0/EMA updates; full native LR warmup, horizon300. Models share initial seed168 and auxiliary checkpoints; only provider/global training streams differ. All32 ENTRY100 snapshots sealed before any held-image diagnostic.

The frozen rule applies only within the original action11/COVERAGE-selected region, for mixed-target→EMA conflicts1→0 or2→1. It exchanges the two relevant probabilities without changing their multiset. Labels enter diagnostic scoring (and the established255 ignore mask), never conflict selection. No prototype computation or support labels. Each of128 held-image/context/stream exposures uses EMA/memory/memory-flip/student forwards; full model/EMA/memory/optimizer/provider/RNG state restored and checked unchanged.

The quality diagnostic provides model-level image holdout. All eight images were already involved in research development, and the class directions were selected from prior development evidence. Therefore this is not researcher-independent confirmation or a new population. It does not query actual hidden-U truth. The proposed five continuous branches were not run after the prerequisite failed; new/old task performance and whether RULE beats IGNORE_C/EMA_ONLY/OFF remain unmeasured here.

## Interpretation and comparison with V122

V122 showed positive target-quality effects on model-fitted A_fit images. Those signs did not carry over to this preregistered, newly rebuilt model-held-out design. This is evidence against advancing the exact frozen conflict rule directly to the proposed five-arm training or RL expansion.

It does not isolate training-image exposure as the sole cause: this design also changes fit membership, the larger annotation condition from n8 to n4, and reconstructs warmup states. It is not a one-factor causal demonstration of overfitting. Fold heterogeneity and four positive context means remain reported; no posthoc fold, direction, confidence threshold or source-state selection replaces the failed gate. The result neither proves every EMA conflict rule fails nor proves that parameter memory should be removed.

## Actual cost, qualification and audit

* Native updates3,224=3,200 entry reconstruction+24 qualification. All attempt/success pairs match,0 failures. Conditional32,000 updates were not spent. No actor optimizer, solve, prototype forward or new annotation.
* Qualification: original selected-loss BASE2 versus implementation BASE2 golden state equality; five methods each two updates and two serialized-checkpoint replay updates. All full-state replays pass; memory remains frozen and EMA/memory have no gradients.
* Diagnostic128 labeled image exposures,512 explicit model-image forwards. No performance query calls, old-task queries, Q_dev, sealed test or original Q_train_new reads. Total recorded forwards including reconstruction/qualification/diagnosis:student21,952; EMA6,576; frozen memory13,152. Do not add the512 diagnostic forwards again to these total forward counts.
* Audit PASS:65 jobs,3,224 physical/rate pairs,24 selector records,512 diagnostic forward pairs,0 performance query pairs; root/child accounting, image exclusions, seal ordering and full readout recomputation checked. The create-only completion audit has already run once and must not be repeated.
* Cumulative native updates844,285 excluding historical source8,000, or852,285 including it. Actor181,593 and linear solves57 unchanged. V119–V123 explicit diagnostic forwards total1,152, labeled diagnostic exposures288; historical512/146 CPU actor-forward diagnostics and all earlier failures remain separately recorded.

A PyTorch warning notes optimizer.step wrapping after scheduler construction. The existing native engine invokes optimizer.step before scheduler.step; this ordering was unchanged. Per-update effective rates were audited. There were no runtime failures or recovery attempts.

## Delivery and next boundary

Publish committed source/protocol, all32 quality rows and all16 context/subgroup summaries, exact costs, qualification, completion audit and this report. Private image IDs, images, labels, per-image details, checkpoints and raw logs remain on NAS. The startup publication is not the final publication; final delivery is confirmed separately by the NAS FINAL_PUBLICATION receipt with remote commit and anonymous HTTP status.

This status closeout starts no next experiment. Preserve the failed prerequisite and the unspent conditional stage. Any future scientifically distinct hypothesis needs a separate frozen protocol and must not silently retry the failed rule or change this gate.
