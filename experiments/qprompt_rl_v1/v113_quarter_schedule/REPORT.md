# V113 complete after metadata recovery: lower rate mitigates, does not reverse decline

The fixed quarter-rate intervention passed the preregistered broad rate-sensitivity criterion. Across all36 paired trajectories, new and old Dice improved by2.538866 and1.493695 percentage points relative to their V1111x controls, and utility improved by0.037343396. Allthree policies nevertheless retained negative mean weightednewgain and negative mean FINALnew−ENTRY100. This is partial mitigation, not beneficial absolute continuation, an RL advantage, or campaign success.

## Complete paired and absolute outcomes

AllDice changes below are percentage points; utility is in original units.

| Policy | New vs1x, pp | Old vs1x, pp | Utility vs1x | FINALnew−entry, pp | Weightednewgain, pp | FINALold−entry, pp |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 2.158057 | 1.178616 | 0.031211549 | -0.846211 | -0.660556 | -0.642577 |
| GLOBAL | 3.059716 | 1.970936 | 0.046272160 | -0.651433 | -0.469843 | -0.578112 |
| TIME | 2.398825 | 1.331533 | 0.034546478 | -0.769548 | -0.603058 | -0.537972 |
| POOLED | 2.538866 | 1.493695 | 0.037343396 | -0.755731 | -0.577819 | -0.586220 |

All36paired rows retain MID/FINALchanges, oldENTRY and memory-reference differences and utility−frozen. The frozen ENTRY100 comparator still beats each policy in mean utility: differences OFF−0.013031324,GLOBAL−0.010479551,TIME−0.011410308. The three-policy pooled FINALnew and FINALold changes from entry are−0.755731pp and−0.586220pp. Five of36individual trajectories improved FINALnew; overallnegative does not mean everytrajectorynegative.

| Context, pooled policies | New vs1x, pp | Old vs1x, pp | FINALnew−entry, pp | FINALold−entry, pp |
|---|---:|---:|---:|---:|
| dev0 | 0.390094 | -0.584042 | -0.052648 | 0.500904 |
| dev1 | 2.475090 | 2.397450 | -0.768870 | -1.144558 |
| dev2 | 5.104102 | 2.325359 | -1.875256 | -1.013193 |
| dev3 | 2.186178 | 1.836013 | -0.326150 | -0.688035 |

Context0 loses oldDice relative to1x while improving oldDice relative toENTRY100; the remainingcontexts improve relative to1x but remain belowentry. Context2 retains the largest new-task decline. These differences are preserved without selecting contexts, policies or streams. Ratesensitivity does not distinguish overfitting, adverse optimization direction, inheritedoptimizer transient or checkpointquality. Nativebody uses GroupNorm, so a BatchNorm-running-stat intervention is not applicable to this model.

## Audit and preserved engineering failure

All25newjobs exited0. All7208native optimizer attempt/success pairs exactly matched root/child ledgers; zero failedmodelcalls. All7208RATErecords satisfy effective=.25×base. There were304stableprobes and108querycalls/432Q_devimageevaluations. The36paired keys/actions/entryreferences and rewardformulas matched, and alltraining completed before the72-snapshotbarrier, with evaluation afterward. Reused V111fit artifacts incurred0newactorupdates/linear solves and were excluded fromnewjob/cost counts.

The original coordinator exited1 after all results were written: it used a create-only writer for COSTS.json, which the scheduler had already created. Original FAILURE.json, EXIT.json and COSTS.json remain untouched onNAS. recover_metadata.py reconstructs completed cost metadata from finishedjob/physicalledgers with0additionalnative/actor/query calls and writes create-only COMPLETED_COSTS.json and METADATA_RECOVERY.json. The independent completionaudit accepts that explicit recovery record and reports original_root_exit_code=1; it does not claim the original run exited0. The source now uses the existing mutable operational writer for final COSTS to prevent recurrence in future runs. Original executed source remains preserved onNAS and at theexecutioncommit.

Cumulative throughV113:776617nativeupdates excluding sharedsource8000 (784617including),181459actorupdates,57linear solves (53data4synthetic). Every prior failure and negative result remains in accounting.

## Next single hypothesis

Keep the quarter-rate schedule fixed. Test one Adam-state restart atENTRY100, without changing weights, scheduler position, actions, data or trajectories. The four actual ENTRY100 checkpoints each have28nonzero first/second-moment parameterstates with Adamstep100. This confirms that the proposed intervention exists; it does not show those states are harmful. Initial100updates used action0, whereas subsequentfixed policies switch toOFF/action11, motivating a bounded optimizer-state diagnostic. Resetting first/secondmoments and Adam bias-correction step is one joint restart intervention; it cannot attribute any effect to momentum alone. Freeze its full36paired matrix and budget before launch, reuse allV113quarter-rate controls, and do not add a rategrid or select an earlystop.

All evidence uses sharedsource/patients and repeatedly inspected developmentdata. Publicdelivery includes source/protocol/allscalarrows/paired/contexttables/costs/recoveryfailure/audit/report; privateimages/features/weights/rawlogs remainNAS.
