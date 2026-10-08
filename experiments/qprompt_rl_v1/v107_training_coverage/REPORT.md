# V107 completed: training-state coverage collection

The frozen training-only collection is complete: 48 new states and 576 action returns, combined with the unchanged V101/V103 32 states and 384 returns into 80 states ×12 actions. This is preparation, not evidence of RL improvement. There were no actor fits, new development reads, new annotation cases or new data sources.

## Completion and engineering recovery

All37 workers exited0. All85,208 native optimizer attempts succeeded:8 qualification,1,200 entry construction and84,000 collection. The coordinator initially exited1 after sealing the dataset because its create-only JSON writer tried to create the existing mutable COSTS.json. The failure happened after collection and dataset validation, not during an optimizer or query call. Original failure/exit/cost records remain unchanged. A metadata-only recovery checked all root/child ledger keys and paired events, all960 return keys/formulas, finite array shapes and exact preservation of old features/returns, then wrote FINAL_COSTS.json and completion markers. Recovery used zero new updates, queries or probes. The public source fixes the mutable cost writer; the original executed source remains identified by the preregistered execution commit.

| Completed operation | Count |
|---|---:|
| Native updates |85,208|
| Actor optimizer updates / linear solves |0 /0|
| Q_train image evaluations |5,664|
| Q_dev image evaluations |0|
| Stable probe extractions |208|
| Original behavior extractions |340|
| Failed optimizer / query calls |0 /0|
| Metadata finalization failures |1, recovered|

Cumulative through V107:608,573 native updates excluding the shared8,000 source updates (616,573 including source),173,265 actor updates and55 linear solves (51 data,4 synthetic). Historical failures and negative results remain counted and published.

## Dataset and interpretation

The12 added contexts are memory steps2000/8000 × labeled support2/8 × identity1,brightness1.2,contrast0.8, each with streams1/2. Existing A_fit, U_adapt and Q_train roles only. Behavior, native trainer,12actions, branch budgets and dense reward were frozen before collection. NEW_RETURN_TABLE.json/csv publishes all576 new aggregated action rows without private features. The new table's raw-return range and full80-state mean per action are recorded in TRAINING_SUMMARY.json. Full-table and both time-specific global choices are action9, so the existing global/time equivalence remains valid. A hindsight training-action maximum is not a deployed policy gain.

V106 failed the practical sequential criterion. This expansion was motivated by its observed feature ranges and deliberately includes previously viewed development photometric factors. It is development-informed and uses the same source/cases; it cannot establish independent patient or domain generalization. No performance gate is evaluated by this dataset-preparation round.

Next: separately preregister V108 with the same V106 matched WARM/CE/RL/analytical-target CE fits, simple controls and real sequential evaluation on streams3/4/5, using the80-state dataset. Compare paired V108 and historical V106 outcomes descriptively for the coverage intervention, while retaining the original within-round RL success gate. Only a qualifying sequential candidate proceeds to separately frozen fresh-stream confirmation.

Source, protocol, complete new aggregate return table, training summary, costs, completion audit and sanitized failure/recovery receipts are public. Private images, feature tensors, checkpoints and raw logs remain on NAS.
