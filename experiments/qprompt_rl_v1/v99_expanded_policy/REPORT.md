# V99 expanded-action policy learning

**NO_V99_EXPANDED_RL_INCREMENTAL_GAIN.** All26jobs completed successfully at 2026-10-08T10:46:52.101077+00:00. Expanded expected-return RL slightly improves the mean over matched expanded CE, but the gain is below the original magnitude screen, reverses for seed602, and both RL seeds underperform their own expanded initialization. The positive V98 finite-grid capacity has not become a learned RL advantage.

| Method | New soft Dice (%) | Simulated old soft Dice (%) | Original utility |
|---|---:|---:|---:|
| EXPANDED_RL | 74.190009 | 81.410084 | -0.065445736 |
| EXPANDED_CE | 74.172202 | 81.383098 | -0.065820113 |
| EXPANDED_INIT | 74.254622 | 81.440875 | -0.064624201 |
| CONTINUE_CE | 74.170248 | 81.392610 | -0.065863258 |
| POOLED_RL | 74.153320 | 81.437957 | -0.065565247 |
| UNIFORM_12 | 73.646964 | 81.094609 | -0.073476563 |
| GLOBAL_12 | 73.761095 | 81.045200 | -0.072922611 |
| TIME_12 | 73.761095 | 81.045200 | -0.072922611 |

## Matched comparisons and interpretation

Against EXPANDED_CE, RL changes new/old by +0.017807/+0.026986 percentage points, utility +0.000374378.

Against EXPANDED_INIT, RL changes new/old by -0.064613/-0.030790 percentage points, utility -0.000821535.

Against CONTINUE_CE, RL changes new/old by +0.019762/+0.017474 percentage points, utility +0.000417523.

Against POOLED_RL, RL changes new/old by +0.036689/-0.027873 percentage points, utility +0.000119511.

Paired RL-minus-CE utility is seed601: +0.000859406, seed602: -0.000110651. Paired RL-minus-initialization is seed601: -0.000244163, seed602: -0.001398906.

Both training-global and training-time controls choose action11 at both stages and perform poorly on development. RL achieves higher normalized and raw training returns than CE (FIT_SUMMARY.csv), but this does not establish generalization. The new-action mass is about56.4% for RL and56.7% for CE. Training advantages clip five new values; the original288 advantages and their normalization remain unchanged. These are descriptive diagnostics, not justification for silently retuning a threshold or dropping a seed.

All204 individual policy/context rows and all arm means remain in DEVELOPMENT_RESULTS.json and ENDPOINT_RESULTS.csv; both seeds, every historical control and every context are retained. Nine-action historical policies had less reward-generation information and are not information/cost-matched to the new twelve-action fits. Expanded CE/RL share384reward rows, initialization, labels, optimizer and1024updates per seed. Expanded initialization has no new fit updates and is an essential control.

## Execution and accounting

Training reused the32original V92 states and288reward rows, added96returns for actions9..11, and verified the original states and behavior exactly. Six actor instances (two initial, two CE and two RL) and training-derived controls were sealed before any selection. All36new action pairs were sealed before matching the frozen grid outcomes. The sixteen V94 policy/state/probability replays passed before any collection, all26jobs exited zero, root/job ledgers match, no failed updates, coordinator ended and lock released. All36cached rows match their selected grid outcomes and168historical rows remain unchanged; the full verdict was independently recomputed.

New physical cost14,408nativeupdates (8qualification+14,400rewardcollection),4,096actorupdates,1,024trainingimageevaluations through256successful query calls. No new development queries or unique images. The36policy readouts reuse432image-evaluation equivalents; these are neither new queries nor independent student replicas. Cumulative campaign through V99 is419,349nativeupdates excluding the common8,000source updates (427,349including),83,149actorupdates. Historical failures/costs are retained.

## Next question and limits

The existing categorical evaluation uses one fixed random action trajectory per policy/context. Before another fit, separately preregister a zero-update readout of the frozen policy's exact two-action expected outcome and deterministic argmax deployment, using the already complete grid. This isolates sampling effects from distribution-level performance without new reward labels or tuning a temperature. Evaluate all six frozen expanded policies and uniform/training-global/time controls, seal probabilities and argmax actions before outcome lookup, and keep the original sampled outcome alongside them. Do not pick a seed, checkpoint, context or deployment temperature after readout.

This is repeatedD1 development with simulated retention, shared query images and no established patient independence. Training uses16memory-fit images, support2/8 from8support images,4old+4new training query images; development uses4old+4new query images. Hidden U labels, sealed tests and real subsequent domains are not accessed. No GRPO or independent cross-domain success is claimed. Old scientific verdicts remain; the user's amendment allows a separately frozen next experiment rather than changing a failed verdict. Private states, weights, image/role identities remain on NAS.
