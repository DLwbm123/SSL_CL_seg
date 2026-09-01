# Fundus Model-Fisher EWC V2 result

Status: `FAIL_EWC_FEASIBILITY` after a valid, complete execution.

All six registered runs completed 13,400 updates and passed their artifact
gates. The single test readout completed 36/36 cells using 612 model forwards,
2,430 images, and zero optimizer steps. Independent zero-model audits reproduced
the arithmetic and verified that the protected input and checkpoint bytes did
not change.

| Registered gate | Observed | Required | Pass |
| --- | ---: | ---: | :---: |
| Mean final Dice improvement | 0.003147 | 0.010000 | No |
| Seeds with positive final Dice improvement | 2 | 2 | Yes |
| Worst per-seed final Dice improvement | -0.080319 | -0.010000 | No |
| Mean BWT improvement | 0.031341 | 0.010000 | Yes |
| Mean incoming Dice improvement | -0.017747 | -0.010000 | No |

The paired final-Dice changes for seeds 0, 1, and 2 were -0.080319, +0.027949,
and +0.061810. Model-Fisher EWC improved mean BWT and was positive on final Dice
for two seeds, but it missed the registered mean-final margin, the worst-seed
safety margin, and the incoming-Dice margin. The result therefore cannot be
reported as EWC feasibility or overall project success.

The registration permits no retry, coefficient change, threshold change,
additional seed, or checkpoint selection after this result. This is a bounded
baseline-feasibility result, not a clinical claim.
