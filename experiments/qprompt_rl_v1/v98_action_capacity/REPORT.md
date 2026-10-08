# V98 expanded memory-mixture capacity

**EXPANDED_ACTION_MARGIN_EXISTS_ON_FROZEN_D1_ONLY.** The added alpha=.75 actions create a feasible practical tradeoff on the frozen development trajectories: 6,202 of 429,981,696 context combinations meet the original screen. The best utility combination is also feasible. This establishes finite-grid capacity, not a learned RL improvement or independent confirmation.

| Reference | New soft Dice (%) | Simulated old soft Dice (%) | Original utility |
|---|---:|---:|---:|
| UNIFORM_ACTION | 73.775267 | 81.289067 | -0.070331457 |
| CONTINUE_CE | 74.170248 | 81.392610 | -0.065863258 |
| POOLED_RL | 74.153320 | 81.437957 | -0.065565247 |
| V97 post-hoc maximum | 74.232346 | 81.637266 | -0.062881215 |
| V98 post-hoc maximum | 74.383492 | 81.705108 | -0.060852100 |

The V98 maximum improves over continued CE by 0.213244 percentage points new and 0.312498 points old, with utility +0.005011159. Against the previous best learned POOLED_RL, the new/old gains are 0.230172/0.267151 points and utility +0.004713147. These comparisons use a development-label-selected reference with a much larger information advantage; they are not fair learned-policy comparisons.

The new-domain margin exceeds the original +.20pp requirement against continued CE by only 0.013244pp. Just 0.001442% of the context combinations pass. Neither that combinatorial fraction nor the very small margin is a probability of success, confidence interval, or significance result. A learned categorical policy may still miss the useful actions.

## What changed and what did not

The only student-action intervention adds alpha=.75 crossed with the same three class weights to the original alpha={0,.25,.5} dictionary. Original actions 0..8 dispatch through their original function, with exact synthetic loss and gradient preservation. New actions 9..11 preserve memory confidence/flip gate, teacher detachment, native loss weighting, A-only adaptation and the native optimizer/lifecycle. All original checkpoints, source and protocol results remain unchanged.

The diagnostic uses the original ENTRY100 states, H300, actions at100/200 and student streams. Original 324 endpoints, 36 midpoint scores and first-action prefixes are reused from V97. New work comprises252 endpoints,12 MID150 and12 prefix snapshots. This is not a new actor, additional random stream or a rerun selected for favorable performance.

| Context | Best V98 actions | New (%) | Old (%) | Utility difference from V97 context maximum |
|---|---|---:|---:|---:|
| dev0 | 10, 10 | 74.559932 | 81.702565 | +0.007640656 |
| dev1 | 0, 6 | 77.144573 | 80.716211 | +0.000000000 |
| dev2 | 2, 0 | 68.649400 | 83.079640 | +0.000000000 |
| dev3 | 9, 11 | 77.180062 | 81.322017 | +0.000475805 |

New actions change the utility-best choices in dev0 and dev3; dev1 and dev2 retain their old best choices. This is descriptive development analysis, not an action lookup table to deploy or a training target. Separate marginal maxima are 74.438575% new and 81.734106% old; they need not be achieved jointly. Mean absolute forgetting for the utility maximum is 0.046036130 relative to the frozen memory reference.

## Verification and physical costs

Completed 2026-10-08T09:36:27.576785+00:00. All nine jobs exited zero, coordinator ended and lock released. The full576 rows contain252 new and324 reused rows; old rows are unchanged, and their original V97 negative feasibility verdict reproduces exactly. All576 gain/penalty/utility calculations, both marginal bounds and the429,981,696-combination decision reproduce. Every new evaluation job started after the single endpoint lock. New actions passed synthetic normalization, gate-off, teacher-detach, finite/empty-mask gradient and invalid-input checks; native qualification passed two exact paired continuations and inference/RNG isolation.

Root and job physical ledgers agree:26,408 successful student updates (8 qualification,26,400 action branches), zero actor updates, zero failures or recovery. New training image-evaluations0; development image-evaluations2,064, verified through516 query calls with paired attempt/success records. No new unique images. The original V97 cost36,008 student updates/2,736 development image-evaluations remains accounted for; cached outcomes do not become independent replicas.

Cumulative campaign through V98:404,941 native updates excluding the common8,000-update source (412,941 including it),79,053 actor updates. All historical costs and negative stages remain in their reports.

## Next experiment and evidence limits

The next question is whether the expanded actions can be selected from training information. A separately preregistered V99 will extend the existing training-only reward table with the three new actions, compare matched twelve-action CE and expected-return RL from the same expanded initialization, and include initial/uniform/training-global/training-time controls. Its targets will use Q_train only. Q_dev grid scores and maximizing sequences must never enter actor fitting or control selection.

Because V97/V98 already contain every two-action student outcome, subsequent frozen policies can be assessed by replaying their decisions on the original ENTRY100 and appropriate saved PREFIX200 states, then retrieving the corresponding cached outcome. Before using that shortcut, V99 must exactly replay all16 V94 CE/RL seed/context action sequences and probabilities from their saved decision traces with zero student updates/queries. Mismatch is an engineering stop, not an excuse to choose a favorable cached row. All new policy actions must be sealed before cached outcome lookup. Evaluation reuse saves computation but supplies no new independent development data.

Human authorization on2026-10-08 removes old scientific gates as automatic blockers of future research; it does not change their historical verdicts. This original screen still reports useful magnitude criteria, and a positive finite grid does not automatically imply paper success. CurrentD1 is repeatedly observed, query images are shared, retention is simulated rather than a real previous domain, and patient independence is not established. Roles remain16 memory-fit images, support2/8 from8 support images,4old+4new training queries and4old+4new development queries; U labels are never read. Model/state tensors, raw data and role identities remain private onNAS. Publication contains full grids, code, plans, receipts, aggregate diagnostics and this report.
