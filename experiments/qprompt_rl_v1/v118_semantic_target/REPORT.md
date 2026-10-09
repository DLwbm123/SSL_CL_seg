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

The intervention failed on this fixed protocol. SEMANTIC is less harmful than ROTATED, but the logged intervention-rate mismatch (detailed below) prevents isolating semantic correctness from perturbation size; the per-batch prototype exchange does not improve the original target. Qualification established BASE matches original model/optimizer/RNG and both new modes replay exactly; the audited run gives no evidence of an execution failure explaining away the negative outcome.

Possible mechanisms include limited patient diversity, feature metric mismatch or incorrect reassignment; these are hypotheses, not measured pseudo-label accuracy. The later scalar audit rules out missing-class/empty-prototype support as the explanation in this run. Hidden U labels were not read, so do not assert the exact error source. Confidence and entropy values remain unchanged by probability exchange, but that invariance does not make the new class assignment reliable. This is inspired by ProDA, not a reproduction of its full algorithm, and cannot refute prototype methods generally.

Do not advance this exact correction into full RL, add epochs, tune thresholds or use the ROTATED comparison to claim success. The current status request closes and delivers this run; it does not launch another experiment. Any subsequent campaign round requires its own single hypothesis and pre-registration under the existing authorization. Historical matched-CE, absolute sequential and complete fresh-stream confirmation gates remain intact.

## Costs and delivery

9612 native updates (12 qualification+9600 main),9610 extra EMA calls/19220image-forwards;384 Q_train calls/1536images(768new/768old);0 Q_dev, actor optimizer, linear solves, new annotations. Entire physical counts, including normal teacher/student forwards, are in COSTS.json. Synthetic import/tensor/readout selfcheck used0 model/query/optimizer. Cumulative841061 native excluding8000source /849061including;181593 actor updates and57solves(53data+4synthetic) unchanged. Previous failures and separate historical512/146CPUactor-forward diagnostics remain charged.

Public delivery includes frozen source, pre-registration, qualification/startup record, every scalar result and negative result, costs and completion audit. Private images, features, full checkpoints and raw logs remain on NAS; they are not in Git. Git publication verification and the committed-file NAS copy are recorded in FINAL_PUBLICATION.json after push. No additional model/query/training call was used for this report or audit.


## Requested posthoc diagnosis — existing scalar ledgers only

All9600 main-update prototype/selection scalar records were aggregated without images, checkpoint loading, model forwards, queries, fitting or optimization. Original pre-registration, readout/gates and costs remain unchanged. Reproducible ledger aggregation source: aggregate_logged_scalars.py (EXEC_RUN points to the existing private run). Anonymous aggregates plus derived comparisons: POSTHOC_ANALYSIS.json. Findings are descriptive unless the paired intervention itself establishes the comparison.

**Damage starts early and accumulates.** At every scheduled checkpoint all16 paired final-new scores are lower for SEMANTIC than BASE. New-score gaps at150/200/250/300 are−0.0053577945/−0.0097665475/−0.0139560872/−0.0156188551. Old-score differences are+0.0009292047/−0.0017387606/−0.0053837416/−0.0067333914. Thus the first readout already has a broad new-score loss, followed by growing new/old loss. This does not look like only a late endpoint accident; it does not establish the unseen trajectory between sampled points or prove an EMA-feedback cause.

**Correction is active and increasingly frequent.** SEMANTIC changes the target argmax on7.9711% of selected pixel-exposures (37,179,838/466,432,200; repeated exposures, not independent pixels). By50-step windows this fraction is5.8076%,7.4871%,8.7089%,9.8946%. These simultaneous trends are consistent with a compounding error hypothesis, but correlation is not causal isolation.

**Missing-class support is not the explanation.** All9600 main updates across three arms have nonzero labeled-pixel support for all three classes and a present prototype. Minimum class supports are255467/16516/4656; no empty-prototype update occurred. Many correlated pixels do not provide many independent labeled patients: prototypes still use the current two-image batch. Class availability and prototype representativeness are different issues.

**The ROTATED control is not matched for intervention size.** SEMANTIC changes7.9711% versus ROTATED99.5653% of selected target argmax labels. Even at identical pre-update100 states (same selected counts and original EMA selected-class counts), the rates are6.5630% versus99.5604%. Therefore the mismatch precedes trajectory drift. SEMANTIC>ROTATED is compatible with simply changing far fewer original labels and does not isolate useful semantic structure. This is a design limitation, not a code failure, and weakens the earlier class-correspondence interpretation. BASE comparison remains valid because it directly tests whether this complete correction intervention helps.

**An explicit semantic shift occurs before learning.** At the same first update, BASE selected target class fractions are79.8408/14.9599/5.1993%; SEMANTIC74.9946/18.9679/6.0375%. Non-background target share increases4.8462percentage points under the same mask. Whole-run shares are79.2994/15.3248/5.3758% for BASE and73.0645/20.3477/6.5878% for SEMANTIC, but those later aggregates compare diverged models. These are target assignments, not correctness, Dice by class, or proof of false-positive foreground; transition directions require a joint table that was not logged.

**Unchanged entropy/confidence does not preserve evidence validity or gradient direction.** target.py chooses only cosine argmax and discards the winning margin. It swaps that class with the original target maximum, rather than estimating how trustworthy the proposed class is. Illustratively[.90,.08,.02] can become[.08,.90,.02]: the same entropy and maximum support a different class. For detached KL target t and softmax student p the logit gradient is proportional to p−t; swapping target mass directly changes the direction even with identical total loss weights. The code's original EMA>.7 selection admission and original-EMA class weights still apply after swapping; neither certifies the new class. Action11's frozen-memory/EMA blended target can also be overwritten by this purely current-label rule. The fraction of changes to memory-backed pixels was not logged, so memory-protection damage is a hypothesis only.

**The feature rule makes an unvalidated modeling assumption.** The existing learned convolutional segmentation head operates on decoder features. Cosine distance to a single per-batch class mean introduces a different decision rule and discards feature norm, within-class multimodality and the prototype margin; native training does not guarantee that rule outperforms the learned head. Geometry conversion follows the parent convention, so the current evidence is not a demonstrated spatial-alignment bug. ProDA motivates prototype-relative likelihood correction and cross-view target structure learning (CVPR2021 abstract); V118's forced probability permutation lacks those mechanisms and is not a reproduction: https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.html . Failure here is about this adaptation, not that paper or semantic features generally.

**Current fixed-recipe selection offers no observed reward headroom.** For each of the16 observed context-stream pairs, the whole-trajectory reward oracle among BASE/SEMANTIC/ROTATED selects BASE. Oracle increment=0 exactly. This excludes an improvement from picking one of these three whole-trajectory recipes on this table; it is not an upper bound for a per-step mixture, unseen trajectories, other actions or RL generally. Combined with V116's nearly uniform learned policy and V117's insufficient three-selector table margin, the next requirement is evidence that an actionable intervention helps, rather than a larger actor.

**Next diagnostic priority (not executed):** assess whether prototype proposals are more reliable than the original target specifically where they disagree, using a separately frozen current-labeled image-held-out diagnostic that never uses a held-out image to build its own prototype. Treat limited labels and reused development evidence explicitly; this is not independent confirmation. Do not automatically run a grid of thresholds/temperatures/epochs or add modules. If the proposal is not better on disagreement pixels, direct target replacement has no foundation; semantic information could instead remain a selection feature. Any future negative control intended to isolate semantics must also account for edited-pixel exposure and reassignment size. A new experiment requires a separate frozen protocol; this analysis launches none.
