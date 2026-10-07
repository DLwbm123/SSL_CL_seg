# Saved trajectory and state coverage

DEPLOYMENT_STATE_200_NOT_AVAILABLE

The frozen deployment writer saves action sequences and step150/300 snapshots, not step200 feature vectors. No replacement state was reconstructed.
Training-branch reward uses entry+25/+100; endpoint utility uses step150/300 from ENTRY100. This is an observed timing difference, not a proved failure mechanism.

| Method | Decision step | Action counts across four development contexts |
|---|---:|---|
| NATIVE | 100 | {0: 4} |
| NATIVE | 200 | {0: 4} |
| FIXED_BEST | 100 | {8: 4} |
| FIXED_BEST | 200 | {8: 4} |
| UNIFORM_ACTION | 100 | {1: 1, 2: 1, 3: 1, 4: 1} |
| UNIFORM_ACTION | 200 | {4: 1, 5: 2, 6: 1} |
| PRE_FROZEN_601 | 100 | {0: 1, 3: 1, 6: 1, 7: 1} |
| PRE_FROZEN_601 | 200 | {3: 1, 4: 1, 6: 1, 7: 1} |
| PRE_FROZEN_602 | 100 | {0: 1, 3: 1, 6: 1, 7: 1} |
| PRE_FROZEN_602 | 200 | {3: 1, 4: 1, 6: 1, 7: 1} |

Formal-policy variation on native training states:

| Seed | Step | Mean L1 distance from within-step mean probability | Maximum distance |
|---|---:|---:|---:|
| 601 | 100 | 0.003161582 | 0.004157665 |
| 601 | 200 | 0.003113326 | 0.004596072 |
| 602 | 100 | 0.001757472 | 0.002326796 |
| 602 | 200 | 0.001762189 | 0.002861705 |

Small within-step variation is descriptive only. It cannot establish that a network uses only progress, nor prove deployment distribution shift.
