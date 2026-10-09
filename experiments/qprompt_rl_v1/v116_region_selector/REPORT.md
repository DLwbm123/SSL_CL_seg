# V116 complete — no reliable regional-selection signal

All 84 development trajectories and 168 snapshots completed. The recovery coordinator and all 27 jobs exited successfully; the qualification job is explicitly reused from the original attempt. Full operation, query, selection-budget, reward and sealing audits passed. Training/evaluation finished at 2026-10-09 06:36:09 UTC (14:36:09 Asia/Shanghai), approximately 62 minutes after recovery launch.

**Decision: NO_RELIABLE_REGION_SELECTION_SIGNAL.** Both actor seeds have slightly higher utility than their paired imitation control, but the gains are below the frozen practical thresholds. Pooled RL is nearly equal to random selection and below the deterministic coverage rule on utility. Both RL seeds fail the absolute-continuation gate. This is a negative result for the frozen recipe, not a general rejection of RL evidence selection.

## Complete development results

Dice is reported on a 0–1 scale; Δ columns compare FINAL300 with the same stored ENTRY100. RL and CE pooled rows average both seeds equally. Other methods have 12 context/stream trajectories; each pooled learned method has 24.

| Method | New Dice | Old Dice | Utility | Δ new vs entry | Δ old vs entry |
|---|---:|---:|---:|---:|---:|
| RL_601 | 0.76123806 | 0.82841982 | -0.03364791 | -0.00579193 | -0.00573568 |
| RL_602 | 0.76145193 | 0.82843958 | -0.03346047 | -0.00557806 | -0.00571592 |
| RL | 0.76134499 | 0.82842970 | -0.03355419 | -0.00568499 | -0.00572580 |
| CE_601 | 0.76123315 | 0.82841822 | -0.03365299 | -0.00579683 | -0.00573729 |
| CE_602 | 0.76128637 | 0.82841356 | -0.03361129 | -0.00574362 | -0.00574195 |
| CE | 0.76125976 | 0.82841589 | -0.03363214 | -0.00577023 | -0.00573962 |
| RANDOM | 0.76129645 | 0.82844090 | -0.03358084 | -0.00573354 | -0.00571461 |
| CONFIDENCE | 0.75826603 | 0.82749982 | -0.03748500 | -0.00876395 | -0.00665568 |
| COVERAGE | 0.76165719 | 0.82829346 | -0.03331171 | -0.00537279 | -0.00586205 |
| GLOBAL_REUSED | 0.76051566 | 0.82837439 | -0.03441126 | -0.00651433 | -0.00578112 |
| OFF_REUSED | 0.75856788 | 0.82772974 | -0.03696303 | -0.00846211 | -0.00642577 |

Pooled RL has positive MID150 new-task change (+0.00146825), but negative FINAL300 change (−0.00568499), weighted new gain (−0.00389668), old FINAL change (−0.00572580), and utility relative to frozen entry (−0.00962249). The complete outcome cannot be described as positive continual learning. Mean FINAL new and old changes are negative in all three evaluated streams. Context dev0 is positive on both, while dev1–3 are negative on both; context means and all individual results are included.

## Frozen comparisons

| Pooled RL minus control | Δ new | Δ old | Δ utility | Utility gate | Practical gate |
|---|---:|---:|---:|---|---|
| CE | +0.00008523 | +0.00001382 | +0.00007795 | False | False |
| RANDOM | +0.00004855 | -0.00001120 | +0.00002665 | False | False |
| CONFIDENCE | +0.00307896 | +0.00092988 | +0.00393081 | True | True |
| COVERAGE | -0.00031220 | +0.00013625 | -0.00024248 | False | False |
| GLOBAL_REUSED | +0.00082934 | +0.00005532 | +0.00085707 | True | False |
| OFF_REUSED | +0.00277712 | +0.00069997 | +0.00340884 | True | True |

The RL−CE utility differences are +0.00000508 for seed601 and +0.00015083 for seed602. Both have the correct sign, but neither establishes a practically meaningful learning benefit. Pooled RL−random utility is only +0.00002665; RL−coverage is −0.00024248. The utility gate requires at least +0.0005 against every required control, together with the frozen Dice tradeoff. Relative gate=False, absolute gate=False, campaign_success=False. No statistical significance or independent replication is claimed.

## Selection and training diagnostics

The following are anonymous class indices from EMA pseudo-label predictions, not ground-truth accuracy estimates. They summarize selected pixels across the 12 development jobs for each method. Different policy states can have different eligible pixel counts; all methods select exactly floor(eligible/2) per image.

| Method | Class0 selected % | Class1 selected % | Class2 selected % |
|---|---:|---:|---:|
| RL_601 | 89.7429 | 7.8869 | 2.3702 |
| RL_602 | 89.9008 | 7.7601 | 2.3391 |
| CE_601 | 89.7724 | 7.8653 | 2.3623 |
| CE_602 | 89.8394 | 7.8059 | 2.3547 |
| RANDOM | 89.8869 | 7.7733 | 2.3397 |
| CONFIDENCE | 99.9972 | 0.0028 | 0.0000 |
| COVERAGE | 79.7151 | 15.6135 | 4.6714 |

Confidence selection concentrates almost entirely on class0; coverage shifts more selected pixels toward classes1/2. RL, CE and random have similar aggregate class shares. These are descriptive associations: they do not prove that class balance causes the performance ordering, that the policy stayed uniform, or that selected pseudo-labels are correct.

Seed601: 32 groups, mean branch Q_train reward 0.00181809, mean within-group reward range 0.00112526, mean RL gradient norm 0.00304978. These positive training rewards do not establish development generalization.
Seed602: 32 groups, mean branch Q_train reward 0.00185037, mean within-group reward range 0.00104994, mean RL gradient norm 0.00218246. These positive training rewards do not establish development generalization.

## Protocol and limits

One selector combines detached regional learning-value features with already-selected class coverage. Existing V113 action11 targets, quarter learning rate, no Adam reset, original batch-global weight normalization and 50% eligible-pixel budget remain unchanged. Seven deployed methods are two RL actors, two CE imitation actors, RANDOM, CONFIDENCE and COVERAGE. Both seeds train 32 groups of four 50-update branches with fixed branch0 retention. One REINFORCE and one winner-branch imitation update are made per group; this is not PPO/GRPO.

CE uses RL-generated trajectories and does not control collection; it is not the historical analytical target-CE control. Full-budget GLOBAL/OFF comparisons include a change in the selection fraction. The 50-step training reward approximates a longer deployment horizon. Shared source/patients and repeatedly viewed development data do not provide independent patient/domain confirmation. No hidden U labels, sealed tests, new patients or new annotations were accessed. All 168 deployment snapshots were sealed before Q_dev readout.

## Engineering failure, recovery and actual costs

The first run stopped after group0 in both learning jobs because a list was passed to the dictionary checkpoint writer. Updated actors were not committed. The user explicitly authorized repair and continuation. The fix uses a dictionary envelope and passed a production writer/reload/receipt regression check without model or optimizer calls. A new create-only run repeated both learning jobs from their original entries and reused the completed qualification. Original failure artifacts and consumed operations remain intact. See REPAIR.md, RECOVERY.json and RECOVERY_COMPLETION.json.

| Accounting scope | Native updates | Actor updates | Query images | Query calls |
|---|---:|---:|---:|---:|
| Original failed attempt, including qualification | 408 | 6 | 112 | 28 |
| New execution in recovery | 29600 | 128 | 4592 | 1148 |
| **Combined V116 physical actual** | **30008** | **134** | **4704** | **1176** |
| Logical successful protocol, includes reused qualification | 29608 | 130 | 4592 | 1148 |

Repeated failed learning overhead is 400 native updates, 4 actor updates and 112 query images / 28 calls. Qualification (8 native, 2 actor) is counted physically once. Recovery has zero optimizer failures; the earlier error was checkpoint persistence. All 29608 logical native/RATE/SELECTION pairs, 130 logical actor pairs, 1148 successful query pairs, 64 groups/256 reward formulas, actor and snapshot barriers, and per-image pixel budgets passed the audit. There are 0 linear solves, 0 stable global probes and 0 new annotations.

Cumulative campaign physical costs through V116: 818641 native updates excluding the historical common source8000 (826641 including it), 181593 actor updates and 57 linear solves (53 data + 4 synthetic). These cumulative totals retain earlier failed attempts.

## Public delivery

Published: executed source and frozen protocol/repair notes; every anonymous scalar trajectory and paired comparison; method/context/stream tables; all training-group branch rewards; selection summaries; costs and audit receipts. Private images, labels, decision features, checkpoints and raw logs remain on NAS. Recovery run: `v116_recovery_20261009T053320Z`; execution commit: `25c4547bb546f9ca555b85713bebc166d6a7588d`.

This round is complete. No additional seed, epoch, threshold sweep or model query was added during readout. Further research requires a separately frozen single hypothesis; the original matched-target/fresh-stream confirmation gates remain in force.

## Subsequent user-requested mechanism diagnosis

The final policies were scored on all saved same-seed training decision states without new image evaluation or training. Mean normalized RL entropy is0.9999899/0.9999972, close to the uniform value1. Thus there is direct evidence of weak final selection preference on these training states; deployment-state entropy was not measured. See DIAGNOSIS.md, POLICY_AUDIT.json and policy_audit.py. This post-hoc diagnostic adds512CPU actor-forward calls,0segmentation forwards,0queries and0optimizer updates; original experiment counts and negative conclusions are unchanged.
