# V127: bring source-memory supervision forward before the large early drift

Human authorization2026-10-09: conduct experiments following the analysis and improve effects. V126 measured a6.509pp fit improvement but5.713pp held decline from SOURCE to ENTRY100. BASE100→300 adds only0.128pp held decline. The revised hypothesis is early preservation of useful frozen source predictions, not more sophisticated conflict deletion or policy fitting. V124/V125 failures and V126 engineering recovery remain unchanged.

## Fixed experiment

Same16 contexts ×2 paired streams3/4, source2000/8000 M_fit-only source, fold0/1 actual n2/n4, brightness.8/contrast1.2, same held4 and old4. No subset exclusion. Restore each new branch to the identical freshly constructed source state and all RNG/provider/optimizer state. Native trainer, supervised LCTX, labels/U, augmentation, Adam, original LR through99 and quarter LR from100, EMA lifecycle, frozen memory all unchanged. Never train on held labels. Historical BASE uses original EMA action0 through99, then the existing action11 COVERAGE mixed-target rule through299; reuse its previously sealed V123/V124/V126 results rather than retraining it.

New branches both run exactly300 updates:

* EARLY_MEMORY: use existing action11 mixed memory/EMA target with the original COVERAGE half-eligible selection from update0. Its target, confidence>.7, class weight(.5,1,1.5), .5 U coefficient and memory flip/agreement machinery are exactly the prior recipe, just activated earlier.
* EARLY_SELECTED_EMA: compute identical mixed target, selector inputs and COVERAGE selection, but replace the detached selected-loss target with detached EMA q only during0–99. From100 onward use exactly EARLY_MEMORY's rule. All model forward routes remain the same. This distinguishes early mixed-target value from changed selection/weighting; once states diverge, selected pixels need not coincide. No new thresholds, coefficients, ramps, optimizer variants or tuning grid.

Also report FROZEN_SOURCE and FROZEN_ENTRY (historical BASE100) as safety controls with zero updates and cached scores repeated at the nominal endpoints. Neither is learning. They are distinct because the first100 updates already lost substantial held Dice. Do not select a control or endpoint from results.

## Readout and frozen criteria

Save100/150/300 for each new trajectory. Seal all64 trajectories/192 snapshots before any performance query. Evaluate new held4 and old4 once per snapshot:384 calls1536 image forwards. Reuse V126 SOURCE, ENTRY100 and BASE150/300 metrics explicitly tagged cached. Publish all480 method×context×stream×time rows,160 trajectories and full subgroup summaries; no hidden selection or independent-data claim.

Primary gains are relative to SOURCE: weightednew=.25*(new150−source)+.75*(new300−source), finalnew=new300−source, oldgain=old300−sourceold, reward=weightednew+oldgain. Separately report own100→300 changes. EARLY_MEMORY must have weightednew>0, finalnew>0 and oldgain≥−.0025 in every fold and stream. Against each BASE, EARLY_SELECTED_EMA, FROZEN_SOURCE and FROZEN_ENTRY require reward difference≥.0005, practical arm ((finalnew difference≥.002 and old difference≥−.0025) OR (old difference≥.005 and finalnew difference≥−.0025)), positive reward difference in each fold and stream and≥12/16 context means. Report each condition separately. Less negative than BASE is mitigation, not absolute learning. Even a pass is researcher-exposed first-domain photometric development; not independent later-domain CL or RL benefit.

## Qualification and budget

CPU synthetic assertions check detach and exact early-only target routing, inherited selection math, absolute gate, matched-control failure and incomplete tables. Before main training reproduce the historical ctx0/stream3 warm100 exactly and assert full snapshot equals sealed V123 ENTRY100;100 charged native updates. Each new branch2 updates plus exact checkpoint replay2:8 updates. Main32×2×300=19200; total cap19308. Zero actor/linear solves/new annotations/Q_dev/test/new datasets. Source checkpoints are reused, no source training. Charge all image forwards; count clean readout directly (it bypasses student hooks), keep attempted/successful native and query records, selector and rate ledgers. Cumulative native at full success895617 excluding historical source8000 /903617 including; actor181593/solves57 unchanged. V126 added2112 diagnostic forwards, prior1152 remain separately accounted.

Fresh create-only NAS root, verified mount/space/write/read and GPU4–7 memory, reliable scheduler/NAS wrapper/neutral argv. No old process termination. Failure stops dependent jobs and preserves all costs; no implicit retry, replacement seed or added budget. Estimated wall time around30minutes, so deliver background startup after qualification and one brief main-job check. Existing authorized hourly monitor can follow this exact run; no new automation. Complete public code/protocol/results/costs/audit/report delivery via proxy after actual completion; images, IDs, checkpoints and raw private logs remain on NAS.
