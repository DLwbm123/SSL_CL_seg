# V97 finite action-sequence feasibility

**NO_ACTION_SEQUENCE_MARGIN_ON_FROZEN_D1_STREAMS.** All 324 endpoints completed; none of the 43,046,721 combinations met the preregistered utility and practical tradeoff criteria. This is a negative result for the fixed finite trajectories, not a claim that reinforcement learning cannot work generally.

## What the exhaustive grid establishes

Four original development contexts each have 81 possible sequences of the existing nine actions at steps 100 and 200. Every sequence starts at its original ENTRY100 and ends at step 300 using the frozen native student stream. Each first-action prefix is shared across its nine second-action branches with complete student, optimizer, EMA, provider and RNG restoration. The physical ledger never rolls back. This enumerates all outcomes reachable by choosing those two actions under these fixed conditions; it makes no assumption that an actor can identify the best sequence without query labels.

All 324 final checkpoints and 36 step150 readouts were sealed before new development scoring. The exact Cartesian calculation returned zero feasible combinations. Original utility is .25(new150-new100)+.75(new300-new100)-max(0,memory_reference-old300-.005). All previous STOP decisions and V84 NOT_RUN remain unchanged.

| Reference | New soft Dice (%) | Old soft Dice (%) | Original utility |
|---|---:|---:|---:|
| UNIFORM_ACTION | 73.775267 | 81.289067 | -0.070331457 |
| CONTINUE_CE | 74.170248 | 81.392610 | -0.065863258 |
| POOLED_RL | 74.153320 | 81.437957 | -0.065565247 |
| Post-hoc grid maximum utility | 74.232346 | 81.637266 | -0.062881215 |

The unconstrained utility maximum exceeds the utility threshold -0.065065247 by 0.002184032, but fails the practical tradeoff requirement. Its sequences are dev0=(8,7), dev1=(0,6), dev2=(2,0), dev3=(6,8). These are diagnostic outcomes, never deployed or supplied to actor training. All 324 channel-wise scores and all 168 historical results remain in ACTION_GRID.csv, GRID_RESULTS.json and REUSED_RESULTS.json; DECISION.json retains all 23 historical arm means.

## Why another actor-only fit cannot clear this frozen screen

The following marginal analysis was added after reading the complete grid and is explicitly exploratory. It uses no new training or label queries. For each metric separately, choose its maximum sequence per context and average the four maxima. These two maxima need not be attained by the same sequence combination.

| Separately maximized score | Maximum (%) | Improvement over CONTINUE_CE (percentage points) | Required improvement |
|---|---:|---:|---:|
| New-domain score | 74.284985 | 0.114737 | 0.20 |
| Simulated old score | 81.653525 | 0.260915 | 0.50 |

Both necessary improvement thresholds are unreachable even before checking the corresponding non-inferiority constraints. The same obstruction holds against POOLED_RL (maximum new/old improvements 0.131665/0.215567 percentage points). Consequently mixtures over these finite outcomes, including expectations over the actor's action sampling with the same student trajectories, also cannot exceed either marginal bound. This is an empirical finite-grid certificate, not a population, unseen-stream, or unseen-domain bound.

Further actor optimization, reward rescaling, fitting seeds, or action-selection rules alone cannot satisfy the original screen on these exact conditions. That conclusion does not exclude smaller real gains: V94 improved over its matched CE and the maximum grid utility is better still. It means the prespecified practical margins require a change beyond actor fitting. The screen must not be relaxed after seeing this result.

## Execution and accounting

Completed 2026-10-08T01:52:12.910930+00:00. All nine jobs exited zero; the coordinator ended and released its lock. All 168 historical context/action-sequence results replay exactly (maximum metric difference 0.0, tolerance 1e-7), including short/new/old channel scores, gain, penalty, utility and absolute forgetting. Feasibility and synthetic threshold/index checks reproduce. Root and job ledgers agree, every attempt has a success, and no failure or recovery occurred. All evaluation start times follow the single seal barrier.

New cost: 36,008 native updates = 8 qualification + 3,600 shared-prefix + 32,400 suffix updates; zero actor updates. Prefix sharing saves 28,800 updates relative to naively retraining all 324 full 200-update sequences. New training-query evaluations: 0; development image-evaluations: 2,736, confirmed in 684 paired query-call ledger records; new unique images: 0. Four-context entry generation and historical actor/reward-generation costs remain prior costs, never reclassified as free new evidence.

Cumulative campaign through V97: 378,533 native updates excluding the common 8,000-update source (386,533 including it), and 79,053 actor updates, including prior failures as previously recorded. No extra random streams, retries, checkpoint search or actor fits were added in V97.

## Limits and next decision

The roles still comprise 16 memory-fit images, an 8-image support pool used at support counts 2/8, four old and four new training-query images, and four old and four new development-query images. Unlabeled pools remain unlabeled; no hidden U labels were read. Repeated evaluations do not create independent samples. Retention is simulated within D1; patient independence and real sequential-domain retention are not established. All development contexts have been repeatedly observed. This is not independent confirmation or paper-ready positive evidence.

The present action dictionary mixes frozen memory predictions with alpha in {0,.25,.5} under the existing gate and three class-weight vectors. A concrete next hypothesis is that this dictionary has insufficient retention capacity. The separately numbered V98 proposal tests only an added alpha=.75 level, retaining the original actions and all other mechanics. Its preregistration is a **proposal awaiting explicit user authorization**, because the current human instruction fixes the existing nine actions. No V98 experiment or changed-action policy is authorized or started. A positive feasibility result would still require separately preregistered, matched CE/RL training and suitable independent confirmation; it would not itself establish RL superiority or unlock V84.

Public delivery includes code, frozen plans, the full anonymized grid, historical controls, replay/ledger/job receipts, decision and this report. Images, labels, identity mappings, student states, model weights and private paths remain on NAS.
