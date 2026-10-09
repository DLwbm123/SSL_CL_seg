# V118 semantic target correction — completed

Decision: **NO_RELIABLE_TRAIN_ONLY_TARGET_REPAIR_SIGNAL**. Semantic correction worsened the original target in all8 context means and both stream means. Relative to BASE, pooled final-new gain decreased0.0156188551, old gain decreased0.0067333914, and signed reward decreased0.0197869814. It is substantially better than deliberately incorrect ROTATED, but that does not establish useful correction versus the original target.

Execution completed2026-10-09 08:27:09UTC /16:27:09Asia-Shanghai. Root and all33 jobs exit0, coordinator lock free, no failure. Completion audit PASS:9612 native/rate/selection records,9610 prototype pairs,384 query pairs/1536images, full192 timepoints/48 trajectories, root-child ledgers, exact pixel budgets, all snapshots sealed before queries, and frozen readout recomputation. No Q_dev, hidden U labels, new patients, actor optimizer or linear solve. Run `v118_semantic_20261009T080849Z`; frozen execution commit `4d04cd7eb3dfb5fbfb179260023293c268f49ddd`. Startup publication `13a31efc2cdf08911573a147a4b9cfe730d031b3`. Source/protocol remain unchanged.

## Complete primary results

All metrics are changes from the cached exact ENTRY100 Q_train scores. `gain`=.25(new150−entry)+.75(new300−entry); `final_new_gain`=new300−entry; `old_gain`=old300−entry; `reward`=gain+old_gain. Scores are fractions, not percentage points. All methods used original COVERAGE selection, identical U mass and matched extra EMA compute.

| method | n | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- |
| BASE | 16 | 0.0021862963 | 0.0025040735 | 0.0050904676 | 0.0072767639 |
| SEMANTIC | 16 | -0.0108672937 | -0.0131147816 | -0.0016429238 | -0.0125102175 |
| ROTATED | 16 | -0.0934863713 | -0.1139340886 | -0.0995736652 | -0.1930600365 |

SEMANTIC−BASE practical gate=False; semantic per-stream absolute gate=False; each-stream advantage over both controls=False; positive context means=0/8. SEMANTIC−ROTATED passes its descriptive comparison but cannot substitute for BASE comparison. Both streams have negative semantic absolute new gain and negative mean old gain. BASE pooled gains and both stream means are positive on Q_train; this is relevant continuous100→300 training evidence, not independent evaluation, actual subsequent-domain continual learning or an RL improvement. BASE itself has negative new context means3/5, and context3 old gain is negative. Shared sources/patients and repeatedly inspected training queries remain development evidence.

Paired semantic reward was below BASE in 16/16 context-stream pairs. Every context mean final-new gain also decreases versus BASE. Old gain is lower in7/8 context means (context7 is the exception), so this is not merely a tolerable retention-for-learning tradeoff.

## Each stream

| stream | method | n | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | BASE | 8 | 0.0030595285 | 0.0031817351 | 0.0051610544 | 0.0082205830 |
| 3 | SEMANTIC | 8 | -0.0105036939 | -0.0130294990 | -0.0014258204 | -0.0119295144 |
| 3 | ROTATED | 8 | -0.0897276964 | -0.1090930612 | -0.0945122489 | -0.1842399453 |
| 4 | BASE | 8 | 0.0013130640 | 0.0018264120 | 0.0050198808 | 0.0063329448 |
| 4 | SEMANTIC | 8 | -0.0112308934 | -0.0132000642 | -0.0018600272 | -0.0130909206 |
| 4 | ROTATED | 8 | -0.0972450462 | -0.1187751160 | -0.1046350815 | -0.2018801277 |

## Every context

| context | method | n | gain | final_new_gain | old_gain | reward |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | BASE | 2 | 0.0034378143 | 0.0033022650 | 0.0093772560 | 0.0128150703 |
| 0 | SEMANTIC | 2 | -0.0061756261 | -0.0080921687 | 0.0037604086 | -0.0024152175 |
| 0 | ROTATED | 2 | -0.1190414336 | -0.1441287510 | -0.1235072892 | -0.2425487228 |
| 1 | BASE | 2 | 0.0043268921 | 0.0047285669 | 0.0064964406 | 0.0108233327 |
| 1 | SEMANTIC | 2 | -0.0161383841 | -0.0197050124 | 0.0058039613 | -0.0103344228 |
| 1 | ROTATED | 2 | -0.1601252567 | -0.1989915026 | -0.1325605819 | -0.2926858386 |
| 2 | BASE | 2 | 0.0041631553 | 0.0038977079 | 0.0012063421 | 0.0053694975 |
| 2 | SEMANTIC | 2 | -0.0026114238 | -0.0040267967 | -0.0071355850 | -0.0097470088 |
| 2 | ROTATED | 2 | -0.0109598553 | -0.0148399211 | -0.0367927216 | -0.0477525769 |
| 3 | BASE | 2 | -0.0006857011 | -0.0012591518 | -0.0087999292 | -0.0094856303 |
| 3 | SEMANTIC | 2 | -0.0096389661 | -0.0116624683 | -0.0156357437 | -0.0252747098 |
| 3 | ROTATED | 2 | -0.0463264594 | -0.0602865256 | -0.0762957577 | -0.1226222171 |
| 4 | BASE | 2 | 0.0012404565 | 0.0025620982 | 0.0066748187 | 0.0079152752 |
| 4 | SEMANTIC | 2 | -0.0178036643 | -0.0203768797 | -0.0137598030 | -0.0315634673 |
| 4 | ROTATED | 2 | -0.1451948783 | -0.1680765953 | -0.1719093807 | -0.3171042590 |
| 5 | BASE | 2 | -0.0070903189 | -0.0068425089 | 0.0094444044 | 0.0023540854 |
| 5 | SEMANTIC | 2 | -0.0335184978 | -0.0390407257 | -0.0011379533 | -0.0346564511 |
| 5 | ROTATED | 2 | -0.1364476308 | -0.1634270921 | -0.1501360033 | -0.2865836341 |
| 6 | BASE | 2 | 0.0073823398 | 0.0080192760 | 0.0050702095 | 0.0124525493 |
| 6 | SEMANTIC | 2 | 0.0013299203 | 0.0007118806 | 0.0014654547 | 0.0027953750 |
| 6 | ROTATED | 2 | -0.0542444931 | -0.0676275939 | -0.0612510778 | -0.1154955709 |
| 7 | BASE | 2 | 0.0047157323 | 0.0056243353 | 0.0112541988 | 0.0159699311 |
| 7 | SEMANTIC | 2 | -0.0023817075 | -0.0027260818 | 0.0134958699 | 0.0111141624 |
| 7 | ROTATED | 2 | -0.0755509632 | -0.0940947272 | -0.0441365093 | -0.1196874725 |

## All scheduled checkpoints (pooled, descriptive)

These are the pre-registered150/200/250/300 readouts; no checkpoint is selected after seeing them. Full192 individual records and48 derived trajectory records are in RESULTS.json/csv and TRAJECTORIES.json.

| method | step | new_gain | old_gain |
| --- | --- | --- | --- |
| BASE | 150 | 0.0012329645 | 0.0057624294 |
| BASE | 200 | 0.0025825175 | 0.0056033586 |
| BASE | 250 | 0.0025144331 | 0.0055063972 |
| BASE | 300 | 0.0025040735 | 0.0050904676 |
| SEMANTIC | 150 | -0.0041248300 | 0.0066916342 |
| SEMANTIC | 200 | -0.0071840300 | 0.0038645980 |
| SEMANTIC | 250 | -0.0114416541 | 0.0001226556 |
| SEMANTIC | 300 | -0.0131147816 | -0.0016429238 |
| ROTATED | 150 | -0.0321432194 | -0.0245605216 |
| ROTATED | 200 | -0.0727166100 | -0.0592488407 |
| ROTATED | 250 | -0.1015342363 | -0.0871845086 |
| ROTATED | 300 | -0.1139340886 | -0.0995736652 |

## Interpretation and boundaries

The intervention failed on this fixed protocol. Class-correspondence matters (SEMANTIC is less harmful than ROTATED), but the tested per-batch prototype exchange does not improve the already available pseudo-target. Qualification established BASE matches original model/optimizer/RNG and both new modes replay exactly; the audited run gives no evidence of an execution failure explaining away the negative outcome.

Possible mechanisms include limited current-label prototype support, coarse feature geometry or incorrect reassignment; these are hypotheses, not measured pseudo-label accuracy. Hidden U labels were not read, so do not assert the exact error source. Confidence and entropy values remain unchanged by probability exchange, but that invariance does not make the new class assignment reliable. This is inspired by ProDA, not a reproduction of its full algorithm, and cannot refute prototype methods generally.

Do not advance this exact correction into full RL, add epochs, tune thresholds or use the ROTATED comparison to claim success. The current status request closes and delivers this run; it does not launch another experiment. Any subsequent campaign round requires its own single hypothesis and pre-registration under the existing authorization. Historical matched-CE, absolute sequential and complete fresh-stream confirmation gates remain intact.

## Costs and delivery

9612 native updates (12 qualification+9600 main),9610 extra EMA calls/19220image-forwards;384 Q_train calls/1536images(768new/768old);0 Q_dev, actor optimizer, linear solves, new annotations. Entire physical counts, including normal teacher/student forwards, are in COSTS.json. Synthetic import/tensor/readout selfcheck used0 model/query/optimizer. Cumulative841061 native excluding8000source /849061including;181593 actor updates and57solves(53data+4synthetic) unchanged. Previous failures and separate historical512/146CPUactor-forward diagnostics remain charged.

Public delivery includes frozen source, pre-registration, qualification/startup record, every scalar result and negative result, costs and completion audit. Private images, features, full checkpoints and raw logs remain on NAS; they are not in Git. Git publication verification and the committed-file NAS copy are recorded in FINAL_PUBLICATION.json after push. No additional model/query/training call was used for this report or audit.
