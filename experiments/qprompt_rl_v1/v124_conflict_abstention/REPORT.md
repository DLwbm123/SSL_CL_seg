# V124 — conflict abstention: complete

**NO_RELIABLE_MODEL_HELD_OUT_ABSTENTION_SIGNAL**. Relative and absolute gates failed. This is a completed negative training experiment, not an engineering failure. IGNORE_C is worse than BASE, RANDOM_MATCHED and UNIFORM_MATCHED on pooled new and old outcomes; it beats OFF. Do not claim useful conflict-location selection or RL benefit.

Finished 2026-10-09T19:42:41.473952+08:00; elapsed2829.8seconds. Root and all65 jobs exited0; once-only completion audit PASS. No runtime failure, retry, restart, new seed or tuning. Source/preregistration commit`3267e68401a8974a63bd0ecf402ec2577c7c2ba3` was proxy-pushed and anonymously verified before launch. Run`v124_abstention_20261009T105359Z`.

## Five complete methods

All changes below are native soft-Dice percentage points relative to the common ENTRY100. Weighted new=.25*(new150−entry)+.75*(new300−entry). Reward is weighted new plus final old change; it is not itself a Dice score. Each method averages all32 condition/stream trajectories equally.

| method | weighted new (pp) | final new (pp) | final old (pp) | reward ×100 |
| --- | --- | --- | --- | --- |
| BASE | -0.10210273 | -0.12815415 | +0.90186978 | +0.79976705 |
| IGNORE_C | -0.21106362 | -0.25355468 | +0.68560170 | +0.47453809 |
| RANDOM_MATCHED | -0.12151964 | -0.14918965 | +0.89074001 | +0.76922036 |
| UNIFORM_MATCHED | -0.10962149 | -0.13649052 | +0.89352406 | +0.78390257 |
| OFF | -0.95579679 | -1.07534452 | -0.56396996 | -1.51976676 |

Every method's pooled weighted and final new changes are negative. BASE and the three partial-removal methods improve old performance on average; a positive reward can therefore hide negative new learning. OFF worsens both pooled new and old outcomes. Absolute benefit must be judged with the separate frozen gate, not signed reward alone.

## IGNORE_C minus each control

| control | final new difference (pp) | old difference (pp) | reward difference ×100 | positive contexts /16 | practical | fold/stream positive | full comparison |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BASE | -0.12540054 | -0.21626807 | -0.32522896 | 3 | False | False | False |
| RANDOM_MATCHED | -0.10436504 | -0.20513830 | -0.29468228 | 3 | False | False | False |
| UNIFORM_MATCHED | -0.11706416 | -0.20792235 | -0.30936448 | 3 | False | False | False |
| OFF | +0.82178984 | +1.24957167 | +1.99430484 | 15 | True | True | True |

The location-specific rule fails against both matched supervision-weight controls, not merely against the no-removal baseline. Only3/16 context means have higher IGNORE_C reward than each of BASE, RANDOM_MATCHED and UNIFORM_MATCHED; no unfavorable condition is removed. Winning against OFF demonstrates relative value of keeping some U supervision in this setting; it does not establish that the chosen conflict locations should be ignored or that new learning is positive.

## Complete subgroup means

Values in percentage points; all positive and negative fold, stream, source, annotation-budget and photometric groups are included. Full16-context rows, all160 trajectories and all320 endpoint rows are in GROUP_SUMMARY.json, TRAJECTORIES.json and RESULTS.json/csv.

| group | method | weighted new | final new | final old | reward ×100 |
| --- | --- | --- | --- | --- | --- |
| fold=0 | BASE | +0.37637753 | +0.37566127 | +1.22135524 | +1.59773277 |
| fold=0 | IGNORE_C | +0.13146964 | +0.09394960 | +0.98478887 | +1.11625852 |
| fold=0 | RANDOM_MATCHED | +0.34283699 | +0.33891473 | +1.21282940 | +1.55566639 |
| fold=0 | UNIFORM_MATCHED | +0.36218784 | +0.35981647 | +1.21262791 | +1.57481575 |
| fold=0 | OFF | -0.90002031 | -1.00578216 | -0.10800567 | -1.00802597 |
| fold=1 | BASE | -0.58058299 | -0.63196956 | +0.58238432 | +0.00180133 |
| fold=1 | IGNORE_C | -0.55359688 | -0.60105897 | +0.38641454 | -0.16718234 |
| fold=1 | RANDOM_MATCHED | -0.58587628 | -0.63729403 | +0.56865062 | -0.01722566 |
| fold=1 | UNIFORM_MATCHED | -0.58143082 | -0.63279751 | +0.57442021 | -0.00701061 |
| fold=1 | OFF | -1.01157328 | -1.14490688 | -1.01993426 | -2.03150754 |
| stream=3 | BASE | -0.20879069 | -0.26109065 | +0.86721359 | +0.65842289 |
| stream=3 | IGNORE_C | -0.31381631 | -0.38218496 | +0.65564769 | +0.34183139 |
| stream=3 | RANDOM_MATCHED | -0.22736679 | -0.28083624 | +0.85771582 | +0.63034903 |
| stream=3 | UNIFORM_MATCHED | -0.21628025 | -0.26907773 | +0.85966508 | +0.64338483 |
| stream=3 | OFF | -1.06551085 | -1.20239530 | -0.57758631 | -1.64309716 |
| stream=4 | BASE | +0.00458523 | +0.00478235 | +0.93652597 | +0.94111120 |
| stream=4 | IGNORE_C | -0.10831093 | -0.12492441 | +0.71555572 | +0.60724479 |
| stream=4 | RANDOM_MATCHED | -0.01567250 | -0.01754306 | +0.92376419 | +0.90809170 |
| stream=4 | UNIFORM_MATCHED | -0.00296273 | -0.00390331 | +0.92738303 | +0.92442030 |
| stream=4 | OFF | -0.84608273 | -0.94829374 | -0.55035362 | -1.39643635 |
| source_step=2000 | BASE | -0.08666053 | -0.12459515 | +0.93686818 | +0.85020765 |
| source_step=2000 | IGNORE_C | -0.27378473 | -0.33929026 | +0.70493114 | +0.43114641 |
| source_step=2000 | RANDOM_MATCHED | -0.11559275 | -0.15649598 | +0.92242607 | +0.80683332 |
| source_step=2000 | UNIFORM_MATCHED | -0.09560981 | -0.13451257 | +0.92839990 | +0.83279009 |
| source_step=2000 | OFF | -0.92873477 | -1.04470033 | -0.55618177 | -1.48491653 |
| source_step=8000 | BASE | -0.11754493 | -0.13171314 | +0.86687137 | +0.74932645 |
| source_step=8000 | IGNORE_C | -0.14834251 | -0.16781911 | +0.66627227 | +0.51792976 |
| source_step=8000 | RANDOM_MATCHED | -0.12744654 | -0.14188332 | +0.85905395 | +0.73160740 |
| source_step=8000 | UNIFORM_MATCHED | -0.12363317 | -0.13846847 | +0.85864821 | +0.73501504 |
| source_step=8000 | OFF | -0.98285882 | -1.10598871 | -0.57175816 | -1.55461698 |
| labeled_images=2 | BASE | +0.40276881 | +0.43643673 | +1.31403371 | +1.71680251 |
| labeled_images=2 | IGNORE_C | +0.21480499 | +0.21700966 | +0.97409934 | +1.18890433 |
| labeled_images=2 | RANDOM_MATCHED | +0.37034346 | +0.40099002 | +1.29134068 | +1.66168414 |
| labeled_images=2 | UNIFORM_MATCHED | +0.38959504 | +0.42193672 | +1.30076818 | +1.69036323 |
| labeled_images=2 | OFF | -1.03640982 | -1.13687531 | -0.78279197 | -1.81920178 |
| labeled_images=4 | BASE | -0.60697427 | -0.69274502 | +0.48970585 | -0.11726842 |
| labeled_images=4 | IGNORE_C | -0.63693222 | -0.72411902 | +0.39710407 | -0.23982815 |
| labeled_images=4 | RANDOM_MATCHED | -0.61338274 | -0.69936931 | +0.49013933 | -0.12324341 |
| labeled_images=4 | UNIFORM_MATCHED | -0.60883802 | -0.69491777 | +0.48627993 | -0.12255809 |
| labeled_images=4 | OFF | -0.87518377 | -1.01381373 | -0.34514796 | -1.22033173 |
| condition=brightness | BASE | -0.29747758 | -0.33192176 | +0.88992245 | +0.59244487 |
| condition=brightness | IGNORE_C | -0.41414637 | -0.46456772 | +0.70335050 | +0.28920413 |
| condition=brightness | RANDOM_MATCHED | -0.30997482 | -0.34511102 | +0.88252872 | +0.57255391 |
| condition=brightness | UNIFORM_MATCHED | -0.30219169 | -0.33685470 | +0.88393376 | +0.58174207 |
| condition=brightness | OFF | -0.94563147 | -1.05513367 | -0.42114444 | -1.36677591 |
| condition=contrast | BASE | +0.09327212 | +0.07561346 | +0.91381711 | +1.00708923 |
| condition=contrast | IGNORE_C | -0.00798087 | -0.04254165 | +0.66785291 | +0.65987204 |
| condition=contrast | RANDOM_MATCHED | +0.06693553 | +0.04673173 | +0.89895129 | +0.96588682 |
| condition=contrast | UNIFORM_MATCHED | +0.08294871 | +0.06387365 | +0.90311435 | +0.98606306 |
| condition=contrast | OFF | -0.96596212 | -1.09555537 | -0.70679549 | -1.67275761 |

## Interpretation and limits

V123 established that forcing EMA's winning class onto these model-held-out conflicts often replaced a correct mixed target. V124 now also finds that discarding supervision there underperforms broad random removal and uniform scaling with the same within-state weight rule. These results reject both tested handling recipes for this particular C; disagreement alone has not supplied a reliable usefulness signal. This does not prove all disagreements are useful, all abstention methods fail, or every learned strategy is impossible.

Matching was exact per-image/EMA-class removal counts for RANDOM_MATCHED and aggregate raw/normalized weight for UNIFORM_MATCHED, at each policy's own evolving state. It did not match KL magnitudes or gradients. Later masks and budgets can differ across divergent trajectories. Random deletion used one predeclared deterministic schedule, not a population over removal seeds. C remains the previously developed1→0/2→1 union, not all EMA-memory conflicts. OFF is supervised continuation after common100-update EMA warmup, not fully supervised from source.

The new evaluation images were excluded from the corresponding source/current fitting, but the eight-image pool was repeatedly used in research development. This is model-level holdout on the first-domain photometric protocol, not independent, multi-domain CL, clinical, deployment or RL confirmation. No Q_dev/test/hidden U labels/original Q_train_new/new annotations were accessed. V123's failed correction gate and unrun Stage B remain unchanged.

## Accounting and delivery

Actual32024 native updates=24 qualifications+32000 main;32024 RATE and selection records;160 trajectories/320 snapshots sealed before704 performance queries2816 labeled image exposures(new1408,old1408). All physical attempts succeeded. No new entry generation, actor updates, linear solves, prototypes, target-quality diagnostic forward or annotation. Full replay/golden BASE/frozen-memory/teacher-gradient qualification passed.

Recorded module-hook image forwards: student256192, EMA64048, memory128096. Source inspection identified that Trainer.clean bypasses the student module hook. The audited2816 performance images therefore add2816 clean student forwards: total student259008 and all-model451152 image forwards. FORWARD_ACCOUNTING.json preserves this source-and-ledger-derived correction; original COSTS.json remains unchanged. No model calls were repeated to recover accounting. Native updates, image reads and model forwards are distinct quantities, not additive budgets. Cumulative native876309 excluding source8000 /884309 including;181593 actor and57 solves unchanged. Prior1152 diagnostic image-forwards288 exposures and512/146 CPU actor-forward diagnostics remain separately recorded. Historical failures and source/entry creation costs are preserved, not reset or recharged.

Public delivery contains committed source/protocol, all anonymous endpoint/trajectory/subgroup/decision/cost/audit tables and this report. Private checkpoints, IDs, images, labels and raw per-update logs remain on canonical NAS. Final proxy push, remote head, anonymous HTTP and NAS FINAL_PUBLICATION receipt must be verified before claiming delivery complete. No completion audit is to be rerun.
