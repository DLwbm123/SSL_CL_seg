# Read-only mechanism summary

## Frozen policy distribution and saved-group gradients

| Domain | Controller | Mean conditional KL to uniform | Mean sampled-set Jaccard | Weighted entropy / policy gradient norm |
|---|---:|---:|---:|---:|
| RIM_ONE_r3 | 601 | 0.00088195 | 0.13308 | 0.3305% |
| RIM_ONE_r3 | 602 | 0.00001062 | 0.15698 | 0.0337% |
| Drishti_GS | 601 | 0.00139388 | 0.15700 | 0.4810% |
| Drishti_GS | 602 | 0.00001586 | 0.15660 | 0.0930% |

These are descriptive averages over the four available states for each policy. Every position and state is retained in the diagnostic JSON. The gradient ratio refers only to the saved last-group behavior policy and averaged four samples; it does not prove the effect of entropy throughout training. No entropy or policy settings were changed.

## Historical actual exposure

| Domain | Method | Controller | Global coverage | Global N_eff | Mean per-window N_eff |
|---|---|---:|---:|---:|---:|
| RIM_ONE_r3 | ALL_U | None | 100.00% | 62.988 | 60.844 |
| RIM_ONE_r3 | GRPO | 601 | 25.40% | 16.000 | 15.974 |
| RIM_ONE_r3 | GRPO | 602 | 25.40% | 16.000 | 15.974 |
| RIM_ONE_r3 | RANDOM | 601 | 100.00% | 56.812 | 15.974 |
| RIM_ONE_r3 | RANDOM | 602 | 100.00% | 56.747 | 15.974 |
| Drishti_GS | ALL_U | None | 100.00% | 40.990 | 40.320 |
| Drishti_GS | GRPO | 601 | 26.83% | 10.998 | 10.925 |
| Drishti_GS | GRPO | 602 | 100.00% | 40.659 | 10.925 |
| Drishti_GS | RANDOM | 601 | 100.00% | 36.741 | 10.925 |
| Drishti_GS | RANDOM | 602 | 100.00% | 35.598 | 10.925 |

Historical global counts are measured; per-window counts are reconstructed from the original loader and verified against the measured totals. Three greedy policies use only the same K-image subset across the full run. This greedy behavior does not imply collapse of their stochastic distributions. Full cumulative coverage also does not imply full-pool diversity in each window.

## Last available branch-0 reward decomposition

| Domain | Controller | Delta labeled quality | Delta weighted source-KL term | Total reward delta |
|---|---:|---:|---:|---:|
| RIM_ONE_r3 | 601 | +0.01447761 | +0.00002654 | +0.01450415 |
| RIM_ONE_r3 | 602 | +0.01357949 | -0.00003024 | +0.01354925 |
| Drishti_GS | 601 | -0.04072165 | +0.00035530 | -0.04036636 |
| Drishti_GS | 602 | -0.02591527 | +0.00009919 | -0.02581607 |

The two terms are additive in reward units, not Dice pp. Other branches and earlier groups cannot be decomposed from missing saved states. Background/rim/cup terms and admitted-pixel fractions are in REWARD_COMPONENT_DIAGNOSTICS.json. The source KL is evaluated on target-domain U, so it does not provide an old-domain retention guarantee.
