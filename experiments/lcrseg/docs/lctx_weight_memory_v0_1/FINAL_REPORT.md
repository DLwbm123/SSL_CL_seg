# LCTX Weight Memory V0.1 — incomplete engineering result

**The fixed experiment did not complete.** Engineering state: `INCOMPLETE_ENGINEERING`. VALUE and SOURCE_GEOMETRY are **not adjudicated**. This report closes the current stopped attempt; it is not a successful or completed scientific matrix.

## Actual execution

Started 2026-09-10T21:41:12.354011+08:00; the last completed training task finished 2026-09-10T22:10:04.667068+08:00. The parent is no longer running. There are18 completed training/evaluation tasks, one failed task, and17 unstarted tasks. Actual formal optimizer/backward/EMA updates:45,920 =45,500 in completed tasks +420 in the failed task, against95,400 planned. Source recovery updates0. All18 training/evaluation exit pairs and their receipt/source/student identities were checked. Available F_FULL trajectories match their parent student/EMA/optimizer/order hashes.

STATIC_SOURCE was not evaluated because not all36 target weights were sealed. No source/target validation reference was substituted. Training U-image reads are0; actual labeled sample accesses91,840 and validation sample accesses1,170. Full qualification costs remain118 synthetic +24 discarded real-smoke updates. The earlier cold-start attempt remains preserved with zero formal updates. No training, new evaluation forward, or GT read was added during this summary.

## Numerical stop and evidence limits

`O1_s62_LR_SRC_AB` (RIM→Drishti, seed62) raised `left constraint residual exceeds 1e-5` at epoch20/update420. The frozen executor stopped new dispatch and allowed already running independent tasks to finish. No tolerance was relaxed and no arm was discarded based on score.

The current saved recovery checkpoint is update399, while the committed step/operation logs reach420: a21-update tail is not represented by the checkpoint. The failing layer/value at420 was not written by the exception handler. Replaying or skipping this tail was not attempted.

A read-only CPU recomputation on the saved399-update checkpoint found the largest rounded left leakage in `bottleneck.block.3.weight`: parameterized-delta leakage 3.84662181e-08, versus leakage of the FP32-added-then-subtracted effective delta 2.08894503e-05. Update norm is 0.00323156591, source weight norm 2.71912684. The former is below1e-5 and the latter is above it. This supports FP32 addition/subtraction roundoff amplified by a small update denominator as a numerical explanation at399; it is not an exact reconstruction of the unsaved420-step CUDA state. Both checks belong to the frozen implementation and were left unchanged.

The checkpoint/basis inspection used CPU tensors only, with no model forward, optimizer update or GT access. This engineering stop does not establish failure of the parameter-protection hypothesis. A future recovery would first need to resolve the numerical contract and account for the unsaved tail; this report does not launch one.

## Descriptive partial results: seed61 only

Only seed61 has all six arms complete in both fixed orders. The following table averages those two orders equally. It does not aggregate the unbalanced collection of18 tasks and does not stand in for three-seed confirmation. Full available per-seed/order/domain/class metrics, including incomplete seed62 coverage, are preserved separately.

| Arm | Final | Incoming | Old | Absolute Forget |
|---|---:|---:|---:|---:|
|F_FULL|0.618883|0.724745|0.513022|0.191045|
|F_CONV|0.624198|0.722043|0.526353|0.177715|
|LR_FREE|0.617087|0.710277|0.523898|0.180170|
|LR_RAND|0.616369|0.713534|0.519203|0.184865|
|LR_SRC_A|0.628510|0.726127|0.530893|0.173175|
|LR_SRC_AB|0.579116|0.682641|0.475591|0.228477|

Final=(Incoming+Old)/2; Forget=matched source score−Old, not clipped. Scores are patient-mean rim/cup macro Dice. No seed SD or full-matrix bootstrap is reported from an incomplete matrix.

| LR_SRC_A minus control, seed61 | Final | Incoming | Old | Forget |
|---|---:|---:|---:|---:|
|F_FULL|+0.009627|+0.001382|+0.017871|-0.017871|
|F_CONV|+0.004312|+0.004084|+0.004540|-0.004540|
|LR_FREE|+0.011423|+0.015851|+0.006995|-0.006995|
|LR_RAND|+0.012141|+0.012593|+0.011690|-0.011690|
|LR_SRC_AB|+0.049394|+0.043486|+0.055302|-0.055302|

The partial seed61 results favor the main candidate on average relative to F_FULL, ordinary low rank and random directions. The dual-side control has lower average Incoming and Old in this complete seed block. These are preliminary descriptive observations; no full VALUE/SOURCE_GEOMETRY success, single-side superiority across seeds, or equivalence claim is made. These results are supervised LCTX adaptation, not SSL confirmation or recovery of KI.

## Delivery and preserved boundaries

RUN_LEDGER.csv covers all36 planned tasks, including the failure and unstarted work. INCOMPLETE_EXECUTION_EVIDENCE.json retains actual counts, completed receipts, memory/merge evidence and the saved-checkpoint CPU probe. PARTIAL_SITE_CLASS_METRICS.csv, AVAILABLE_TRAJECTORIES.csv, AVAILABLE_PAIRED_EFFECTS.csv and AVAILABLE_CLASS_COSTS.csv retain every available aggregate. SEED61_COMPLETE_BLOCK_SUMMARY.csv defines the narrow complete-block summary above. Original source/configuration freeze, failed launch, completed weights and old experiment evidence are unchanged.

Private weights, patient data and raw logs remain on NAS. Only aggregate evidence and engineering reports are public. No main merge, new seed/domain, changed rank/k/LR, additional ASM/AMS run or process restart was performed. The experiment is stopped and incomplete; scientific final reporting remains unavailable.
