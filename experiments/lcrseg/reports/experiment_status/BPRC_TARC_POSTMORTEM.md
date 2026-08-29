# BPRC V0.1 TARC pairwise-geometry post-mortem

**Status:** `BPRC_TARC_POSTMORTEM_COMPLETE`  
**Optimizer steps:** `0`  
**Hidden-GT training usage:** `none`

The analysis used the frozen TARC R0 checkpoints, anchor views, validation case lists, relation temperature, and exact TARC margin/fidelity functions. It did not change the preregistered BPRC formula or gate.

## Findings

- Class-transport mean off-diagonal Gram distortion: `0.313718`.
- Static mean disc-rim margin agreement: `0.773684`.
- Class-transport mean disc-rim margin agreement: `0.764289`.
- Class-minus-static disc-rim margin agreement: `-0.009394`.
- Correlation between class-anchor Gram distortion and disc-rim margin delta: `-0.205786`.

The post-mortem is descriptive only. No transport repair, feature mapping, threshold change, or BPRC formula adjustment was made.
