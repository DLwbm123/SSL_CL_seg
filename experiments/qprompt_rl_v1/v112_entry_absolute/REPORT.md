# V112 complete: continuing adaptation loses to frozen ENTRY100

The16-image entry-old diagnostic completed with0optimizer updates. Every one of the22 V111 methods has negative mean weightednewgain and negative mean FINALnew−ENTRY100. All also have negative mean oldFINAL−ENTRYold and utility−frozenENTRY. This admits the separately frozen V113 quarter-rate diagnostic. It does not identify learning rate as the cause, change the V111 gate or establish frozen-entry continual-learning success.

## Entry references

| Context | ENTRY100 new | ENTRY100 old | Old memory reference | Frozen utility |
|---|---:|---:|---:|---:|
| dev0 | 0.727719814 | 0.795111328 | 0.851620317 | -0.051508988 |
| dev1 | 0.805338427 | 0.843119986 | 0.851620317 | -0.003500330 |
| dev2 | 0.735533245 | 0.861607306 | 0.874554105 | -0.007946799 |
| dev3 | 0.799528465 | 0.836783402 | 0.874554105 | -0.032770703 |

Equal-context mean ENTRY100 new/old is 0.767029988/0.834155506. Mean memory-old reference is 0.863087211. Their difference already includes loss incurred before the decision to continue at step100; it must not be called subsequent forgetting.

## Absolute changes after continuing to step300

Dice changes are percentage points. Sampled learned arms average both predeclared actorseeds equally. Every method/seed/context/stream, secondaryargmax arm and analogous V108 result is retained in the accompanying504-row tables.

| Method | FINAL new−entry, pp | MID new−entry, pp | Weighted new gain, pp | FINAL old−entry, pp | Utility−frozen |
|---|---:|---:|---:|---:|---:|
| GLOBAL | -3.711149 | -1.371048 | -3.126124 | -2.549048 | -0.056751711 |
| NN | -2.185542 | -0.990496 | -1.886781 | -1.522752 | -0.034095325 |
| OFF | -3.004268 | -1.399574 | -2.603095 | -1.821193 | -0.044242873 |
| RIDGE | -1.977115 | -0.990496 | -1.730460 | -1.527310 | -0.032577699 |
| SAMPLE_CE | -2.783993 | -1.059040 | -2.352755 | -1.806446 | -0.041592012 |
| SAMPLE_DISTILL | -2.662579 | -1.059040 | -2.261695 | -1.770811 | -0.040325053 |
| SAMPLE_RL | -2.662579 | -1.059040 | -2.261695 | -1.770811 | -0.040325053 |
| SAMPLE_WARM | -2.771205 | -1.152002 | -2.366405 | -1.849553 | -0.042159579 |
| TIME | -3.168373 | -1.399574 | -2.726173 | -1.869506 | -0.045956786 |
| UNIFORM | -3.836134 | -1.372928 | -3.220332 | -2.379916 | -0.056002486 |

For sampled RL, the4.663981pp old-memory drop decomposes into2.893171pp already present at ENTRY100 and1.770811pp subsequent old-task decline. Its weightednewgain is−2.261695pp and FINALnew change−2.662579pp. Mean originalutility is−0.064256759 versus frozenentry−0.023931705. Thus the better-than-some-controls result does not establish beneficial continued adaptation.

| RL context | FINAL new−entry, pp | Weighted new gain, pp | FINAL old−entry, pp |
|---|---:|---:|---:|
| dev0 | 1.686746 | 1.515669 | 1.750684 |
| dev1 | -2.746963 | -2.416564 | -3.191618 |
| dev2 | -7.220598 | -6.091950 | -3.068613 |
| dev3 | -2.369502 | -2.053934 | -2.573695 |

Negative overall means do not mean every context or individual trajectory is negative. The complete context tables preserve that distinction.

## Selected action sequences

| Sampled RL sequence | Count | FINAL new−entry, pp | FINAL old−entry, pp |
|---|---:|---:|---:|
| OFF->OFF | 9 | -2.895747 | -2.921293 |
| OFF->ON | 5 | -2.915114 | -2.495934 |
| ON->OFF | 0 | NA | NA |
| ON->ON | 10 | -2.326461 | -0.372815 |

Allfour categories, including zero-count groups, are reported per method in SEQUENCE_SUMMARY. These are policy-selected groups confounded with context and previous state; they are not randomized causal OFF→ON comparisons.

## Audit, limits and next run

Four old-query attempt/success pairs and exactly16 imageevaluations were audited. All four queries preserved fullstudent/provider/RNG snapshots and used only existingQ_dev_old identities. All504 reused source rows and absolute formulas were checked; no native/actor/linear/probe/U/newannotation operation occurred. Allprevious query exposure remains development evidence. Cumulative optimizer totals remain769,409 native excluding8,000 source (777,409 including),181,459 actorupdates and57 solves.

V111 was fully published before this query. The standalone5-second diagnostic finished before the subsequent live ps/nvidia inspection, so that inspection found no active process; no rerun was performed. The launch used the same neutral argv/NASwrapper and verified mount,writeprobe andGPU admission. This is a process-observation limitation, not missing experiment outputs.

Proceed with V113 at the one fixed quarter-rate setting, OFF/GLOBAL/TIME×4contexts×3streams,7,200deployment+8qualification nativeupdates. Reuse the matchedV1111x outcomes,432newQ_devimageevaluations,zeroactorfit. No rategrid, seedselection or earlystop. Relative improvement without positive absolute continuation remains partial mitigation.

Public delivery includes frozen protocol, source, all504 scalarrows, entryreferences, complete method/context/sequence summaries, costs,audit and this report. Private images/features/weights/rawlogs remain onNAS.
