# P1 confirmation — awaiting scientific review

Local reports only. Public GitHub publication has not been verified by this exporter.
Primary: seeds 163/164; supplementary descriptive: 162–164. Two orders average within seed first.
These are development patients, not independent-patient generalization, original KI, or SOTA evidence.

| Method | Seed | Order | Origin | Final | Old | Incoming | Forget |
|---|---:|---:|---|---:|---:|---:|---:|
| B0_PARENT_LCTX | 163 | 1 | new_execution | 0.670473 | 0.644593 | 0.722232 | 0.104215 |
| B0_PARENT_LCTX | 163 | 2 | new_execution | 0.626147 | 0.599374 | 0.679695 | 0.174324 |
| B0_PARENT_LCTX | 164 | 1 | new_execution | 0.655883 | 0.632986 | 0.701676 | 0.125771 |
| B0_PARENT_LCTX | 164 | 2 | new_execution | 0.639660 | 0.610454 | 0.698071 | 0.170361 |
| B2_PARENT_PAS_KL | 162 | 1 | new_execution | 0.644670 | 0.598196 | 0.737619 | 0.153427 |
| B2_PARENT_PAS_KL | 162 | 2 | new_execution | 0.639674 | 0.610153 | 0.698717 | 0.159279 |
| B2_PARENT_PAS_KL | 163 | 1 | new_execution | 0.668373 | 0.640572 | 0.723975 | 0.123198 |
| B2_PARENT_PAS_KL | 163 | 2 | new_execution | 0.646559 | 0.613232 | 0.713213 | 0.167174 |
| B2_PARENT_PAS_KL | 164 | 1 | new_execution | 0.705631 | 0.709358 | 0.698178 | 0.038140 |
| B2_PARENT_PAS_KL | 164 | 2 | new_execution | 0.695481 | 0.686479 | 0.713483 | 0.099514 |
| F5 | 163 | 1 | new_execution | 0.705548 | 0.692793 | 0.731056 | 0.064306 |
| F5 | 163 | 2 | new_execution | 0.622870 | 0.594753 | 0.679104 | 0.181903 |
| F5 | 164 | 1 | new_execution | 0.653930 | 0.637226 | 0.687339 | 0.120700 |
| F5 | 164 | 2 | new_execution | 0.634089 | 0.601982 | 0.698302 | 0.180118 |
| B0_PARENT_LCTX | 162 | 1 | historical_import | 0.636851 | 0.590288 | 0.729979 | 0.167161 |
| B0_PARENT_LCTX | 162 | 2 | historical_import | 0.612573 | 0.576782 | 0.684156 | 0.196028 |
| F5 | 162 | 1 | historical_import | 0.658965 | 0.617127 | 0.742639 | 0.136498 |
| F5 | 162 | 2 | historical_import | 0.618365 | 0.575391 | 0.704314 | 0.204064 |

## Frozen budget decisions (not significance tests)

{'G1': False, 'G2': False, 'G3': False, 'order_tradeoffs': {'1': {'Final': 0.01656132809827282, 'Old': 0.026220047843996674, 'Incoming': -0.002756111393174887, 'Forget': -0.02248993099523547}, '2': {'Final': -0.00442421279201094, 'Old': -0.006546164797521015, 'Incoming': -0.00018030878099084546, 'Forget': 0.008668248061138556}}, 'recommendation': 'STOP_NO_ADDITIONAL_EXPERIMENTS'}

All positive and negative seed/order outcomes are retained in PAIRED_COMPARISONS.csv. Passing only recommends P2 review; no P2 runs.

## Cost and integrity

28 new targets, 8 historical target imports, 3 reused sources, zero new source updates.
Worker session seconds include evaluation and shared-GPU waiting; they are not exclusive GPU compute or evidence of speed advantage.
Trainer telemetry and operation_counts overlap; never sum them as independent operations.
Formal, source verification, model integrity, CUDA qualification, smoke, failed and resumed sessions are separate in COST_AND_COMPLETION.json.
All 28 target files have code/plan/node-bound tensor and file integrity evidence; metadata freshness was checked for this export.
