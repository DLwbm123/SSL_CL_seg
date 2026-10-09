# V117 completed: no reliable train-only macro-selection signal

Run `v117_macro_20261009T070553Z` finished2026-10-09 07:25:03UTC /15:25:03Asia/Shanghai, about16m45s after launch. Root and all9 jobs exited0; lock is free. Full completion audit PASS. Execution/preregistration commit `a5fe4173e9e297a0fd2fd5ac0692229731888482`; startup publication `05c8a4f1ce8c9a8cf6c5504d47014da91c22f386`. No repair, retry, hyperparameter change or additional experiment occurred during completion readout.

## Result and meaning

The frozen gate is `NO_RELIABLE_TRAIN_ONLY_MACRO_SIGNAL`. Both NN_LOCO and the outcome-informed oracle pass the short-branch absolute criterion, but this does not establish positive sequential continuation. Every point begins at one of the64 saved V116 training states at steps100/150/200/250 and runs only50steps. This is not a retained new trajectory from ENTRY100 to300, and it does not supersede V116's negative deployment results.

Coverage is the best fixed rule by mean signed reward. GLOBAL_LOCO selects coverage in every held-out state. NN_LOCO is worse than this rule by0.0005455234495457262 reward, with FINALnew higher by0.0002765919780358672 but old retention lower by0.0007511358708143234. Both source seeds have lower reward than their fold-global control, and only2/8 context folds are strictly positive. NN is essentially tied with RANDOM (+0.00004407056258060038 reward); every predeclared NN practical comparison fails.

Even the oracle that knows each held-out state's measured outcomes improves reward over best fixed by only0.00048673516721464694, below0.0005; do not round this into a pass. Its FINALnew increment0.0006285844137892127 and old increment−0.00001860933844000101 also fail the practical tradeoff. Thus this particular three-rule action set has limited measured reward headroom under the fixed50step objective. This is a table-specific diagnostic, not a population upper bound or proof that more expressive selection cannot work. The observed nearest-neighbor readout does not recover reliable conditional benefit; it does not prove every learner would fail.

The results argue against immediately spending another full RL/deployment run on these same three macro-actions. They do not isolate entropy, actor-update count or temporal credit assignment as the cause of V116's failure. Broader action/target quality hypotheses require separate registration; none was launched in this status closeout.

## Frozen design and evidence boundary

See [PREREG.md](PREREG.md). Reused64 complete V116 GROUP_ENTRY snapshots, two source actor seeds×8contexts×4cycles. Each state has paired RANDOM/CONFIDENCE/COVERAGE/OFF branches with50 native updates and restored full native/provider/global RNG state. OFF is a comparator only; NN and oracle choose among the three unchanged half-eligible regional rules. Original V113 quarter rate/no optimizer reset, action11 targets and full original U mass remain fixed; OFF is established action12 with no U forward/loss.

State features are first-minibatch existing prediction statistics (13-dimensional tile feature mean/std,26dimensions); exact feature equality across three paired selector entries was asserted. No query labels/scores/context IDs are predictor features. Eight leave-whole-context-out folds exclude both source seeds and allcycles for the held-out context; training-fold-only scaling and fixed1NN with exact-distance tie averaging. GLOBAL_LOCO uses the training-fold mean. No fit/tuning/actor optimization or Q_dev readout. Oracle and BEST_FIXED_POSTHOC explicitly use outcomes and cannot be described as deployable validation.

All source/patients/Q_train cases have been repeatedly studied. Cycle0 shares the same student entries across source seeds; seeds are not independent replications. Positive short-branch training gains cannot be called independent generalization, RL benefit, or continual-learning success.

## Complete pooled results

All values below are raw Dice-scale differences, not percentage points. `gain`=.25*(new13−entrynew)+.75*(new50−entrynew); `final_new_gain`=new50−entrynew; `old_gain`=old50−entryold; `reward`=gain+old_gain.

| policy | n | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- |
| NN_LOCO | 64 | 0.0008408192 | 0.0009803706 | 0.0011502273 | 0.0019910464 |
| GLOBAL_LOCO | 64 | 0.0006352068 | 0.0007037786 | 0.0019013631 | 0.0025365699 |
| ORACLE | 64 | 0.0011405513 | 0.0013323630 | 0.0018827538 | 0.0030233051 |
| BEST_FIXED_POSTHOC | 64 | 0.0006352068 | 0.0007037786 | 0.0019013631 | 0.0025365699 |
| RANDOM | 64 | 0.0010977432 | 0.0012942306 | 0.0008492327 | 0.0019469759 |
| CONFIDENCE | 64 | 0.0006585953 | 0.0007772795 | -0.0014859736 | -0.0008273783 |
| COVERAGE | 64 | 0.0006352068 | 0.0007037786 | 0.0019013631 | 0.0025365699 |
| OFF | 64 | 0.0005013327 | 0.0005784363 | -0.0014980724 | -0.0009967397 |

## Every context and both source seeds

The complete eight-policy tables are in CONTEXT_SUMMARY.json and SEED_SUMMARY.json. NN, fold-global and oracle are shown for every context below; no context is omitted.

| context | policy | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- |
| 0 | NN_LOCO | 0.0013632514 | 0.0014450327 | 0.0009109825 | 0.0022742338 |
| 0 | GLOBAL_LOCO | 0.0009397555 | 0.0007940475 | 0.0029155323 | 0.0038552878 |
| 0 | ORACLE | 0.0014084387 | 0.0013798131 | 0.0026184907 | 0.0040269294 |
| 1 | NN_LOCO | 0.0010278772 | 0.0011278195 | -0.0010140864 | 0.0000137908 |
| 1 | GLOBAL_LOCO | 0.0014928761 | 0.0016519455 | 0.0026954152 | 0.0041882913 |
| 1 | ORACLE | 0.0014928761 | 0.0016519455 | 0.0026954152 | 0.0041882913 |
| 2 | NN_LOCO | 0.0014126454 | 0.0015820758 | 0.0011163382 | 0.0025289836 |
| 2 | GLOBAL_LOCO | 0.0008938690 | 0.0009549335 | 0.0006574150 | 0.0015512840 |
| 2 | ORACLE | 0.0025279536 | 0.0030281721 | 0.0016361084 | 0.0041640620 |
| 3 | NN_LOCO | 0.0005329296 | 0.0006181011 | -0.0033427617 | -0.0028098321 |
| 3 | GLOBAL_LOCO | -0.0006598181 | -0.0008816486 | -0.0027423240 | -0.0034021421 |
| 3 | ORACLE | 0.0011256568 | 0.0013244962 | -0.0034251781 | -0.0022995214 |
| 4 | NN_LOCO | 0.0005470584 | 0.0006212555 | 0.0027027037 | 0.0032497621 |
| 4 | GLOBAL_LOCO | 0.0005470584 | 0.0006212555 | 0.0027027037 | 0.0032497621 |
| 4 | ORACLE | 0.0005470584 | 0.0006212555 | 0.0027027037 | 0.0032497621 |
| 5 | NN_LOCO | -0.0018440192 | -0.0020037210 | 0.0035955617 | 0.0017515426 |
| 5 | GLOBAL_LOCO | -0.0018440192 | -0.0020037210 | 0.0035955617 | 0.0017515426 |
| 5 | ORACLE | -0.0017062428 | -0.0018624617 | 0.0034625363 | 0.0017562935 |
| 6 | NN_LOCO | 0.0020390821 | 0.0024810983 | 0.0019443883 | 0.0039834704 |
| 6 | GLOBAL_LOCO | 0.0020390821 | 0.0024810983 | 0.0019443883 | 0.0039834704 |
| 6 | ORACLE | 0.0020558191 | 0.0025033653 | 0.0019297414 | 0.0039855605 |
| 7 | NN_LOCO | 0.0016477285 | 0.0019713026 | 0.0032886919 | 0.0049364204 |
| 7 | GLOBAL_LOCO | 0.0016728502 | 0.0020123180 | 0.0034422129 | 0.0051150632 |
| 7 | ORACLE | 0.0016728502 | 0.0020123180 | 0.0034422129 | 0.0051150632 |

| seed | policy | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- |
| 601 | NN_LOCO | 0.0008584829 | 0.0009999918 | 0.0011320901 | 0.0019905730 |
| 601 | GLOBAL_LOCO | 0.0006641590 | 0.0007401551 | 0.0018954317 | 0.0025595906 |
| 601 | ORACLE | 0.0012448455 | 0.0014584458 | 0.0018221959 | 0.0030670414 |
| 601 | BEST_FIXED_POSTHOC | 0.0006641590 | 0.0007401551 | 0.0018954317 | 0.0025595906 |
| 601 | RANDOM | 0.0011660537 | 0.0013725357 | 0.0009190096 | 0.0020850633 |
| 601 | CONFIDENCE | 0.0006724244 | 0.0007947162 | -0.0015005909 | -0.0008281666 |
| 601 | COVERAGE | 0.0006641590 | 0.0007401551 | 0.0018954317 | 0.0025595906 |
| 601 | OFF | 0.0005146912 | 0.0005953815 | -0.0015120257 | -0.0009973345 |
| 602 | NN_LOCO | 0.0008231555 | 0.0009607493 | 0.0011683644 | 0.0019915199 |
| 602 | GLOBAL_LOCO | 0.0006062546 | 0.0006674021 | 0.0019072946 | 0.0025135492 |
| 602 | ORACLE | 0.0010362571 | 0.0012062802 | 0.0019433117 | 0.0029795687 |
| 602 | BEST_FIXED_POSTHOC | 0.0006062546 | 0.0006674021 | 0.0019072946 | 0.0025135492 |
| 602 | RANDOM | 0.0010294328 | 0.0012159254 | 0.0007794558 | 0.0018088885 |
| 602 | CONFIDENCE | 0.0006447662 | 0.0007598428 | -0.0014713563 | -0.0008265901 |
| 602 | COVERAGE | 0.0006062546 | 0.0006674021 | 0.0019072946 | 0.0025135492 |
| 602 | OFF | 0.0004879742 | 0.0005614911 | -0.0014841191 | -0.0009961449 |

## Selection and accounting

Aggregate selected-rule counts (64states each):

- NN_LOCO: CONFIDENCE=16, COVERAGE=42, RANDOM=6
- GLOBAL_LOCO: COVERAGE=64
- ORACLE: RANDOM=6, COVERAGE=44, CONFIDENCE=14

All256 branch rows are published in RESULTS.json/csv; all512 selected-policy rows are in SELECTED_RESULTS.json. DECISION.json includes every unrounded comparison. No hidden-label interpretation of pseudo-class coverage is made.

Physical audit:12808 native optimizer attempts/successes (8qualification+12800collection),12808 quarter-rate records,9604 exact-half-budget selector records,896 Q_train query pairs/3584images (2304new+1280old), all256 reward formulas, exact root/child ledger accounting, all9 exit0, and recomputed LOCO outputs PASS. No failure, actor optimizer, linear solve, Q_dev query, new annotation or new data. Public COSTS.json preserves all counted image forwards and selection operations. CPU qualification selfchecks had146 actor forwards,0segmentation forwards/queries/optimizer calls, separately documented in SOURCE_SELFCHECK.json; first73 were reconstructed by an instrumented deterministic rerun with original receipt retained.

Cumulative native updates831449 excluding8000 shared historical source (839449 including); actor optimizer181593 and linear solves57(53data+4synthetic) unchanged. All previous failed execution costs remain charged. V116 posthoc512CPU actor forwards remain separate from V117's146CPU selfcheck forwards. This audit and report add zero model/query/optimizer calls.

Published scope: committed source/protocol, all anonymous branch/policy/context/seed scalar results, decision, costs, audit and report. Private features, source state snapshots, data and raw operational logs remain on NAS. Original matched-CE, absolute sequential and complete fresh-stream campaign gates remain unchanged; campaign success is false.
