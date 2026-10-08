# V106 stable-state sequential deployment

**V106_NO_PRACTICAL_SEQUENTIAL_RL_GAIN.** Stable-state RL does not outperform continued original CE on this real sequential deployment matrix. Both actor seeds lose to CE; RL is only marginally better than target-matched distillation, with seed602 exactly tied. The complete negative result is retained.

## Primary sampled deployment

| Policy | New soft Dice (%) | Simulated old soft Dice (%) | Utility |
|---|---:|---:|---:|
| SAMPLE_WARM | 73.144152 | 80.994444 | -0.078355046 |
| SAMPLE_CE | 73.198655 | 81.060877 | -0.077188559 |
| SAMPLE_RL | 73.104740 | 80.936650 | -0.079345298 |
| SAMPLE_DISTILL | 73.098750 | 80.931635 | -0.079460410 |
| UNIFORM | 72.819661 | 81.006545 | -0.080632810 |
| GLOBAL | 72.991850 | 80.866503 | -0.080683416 |
| RIDGE | 73.151890 | 80.967283 | -0.078496529 |
| NN | 73.179822 | 81.027941 | -0.077167600 |

RL minus CE is -0.093915 percentage points new and -0.124227 points old, utility -0.002156739. RL minus matched-target DISTILL is only +0.005990 points new and +0.005015 points old, utility +0.000115112. RL exceeds uniform and global policies but trails WARM, CE, ridge and1NN. Only the tradeoff against uniform passes; the full frozen practical criterion fails.

| Actor seed | RL minus CE utility | RL minus DISTILL utility | RL minus WARM utility |
|---|---:|---:|---:|
| 601 | -0.002365567 | +0.000230225 | -0.001229475 |
| 602 | -0.001947910 | +0.000000000 | -0.000751028 |

All six actor-seed/student-stream RL-minus-CE mean utility comparisons are negative. RL and DISTILL select matching action pairs for10/12 context-stream cases at seed601 and12/12 at seed602; residual differences are not a consistent RL-specific advantage. All paired stream values, actions and outcomes are retained.

## Fixed argmax secondary readout

| Policy | New soft Dice (%) | Simulated old soft Dice (%) | Utility |
|---|---:|---:|---:|
| ARGMAX_WARM | 73.062487 | 80.843114 | -0.080661674 |
| ARGMAX_CE | 73.191987 | 80.946979 | -0.078384326 |
| ARGMAX_RL | 73.031692 | 80.815630 | -0.081187244 |
| ARGMAX_DISTILL | 73.036934 | 80.822273 | -0.081085783 |

Argmax does not rescue the result: mean RL is below original CE, matched-target DISTILL and WARM. It remains secondary and does not replace sampled deployment.

## What was executed

All32 existing V101 training states/384 Q_train returns and V103 stable mean4 features trained fresh full-table policies. Seeds601/602 each used512 CE warmup, then copied WARM models received512 matched CE, RL or target-matched DISTILL updates. The12 actions, network, optimizer, reward transform, KL and entropy settings were fixed. Full-table ridge and1NN were trained only from the same training values; time and global choices were verified equal. No development information entered fitting.

Four development conditions and three preregistered future student random streams3/4/5 evaluated20 policies each:8 sampled actors,8 argmax actors and4 simple controls. All240 trajectories performed200 native updates from the shared original ENTRY100 checkpoints, with decisions at100/200 and snapshots at150/300. They are actual student trajectories, not finite-grid lookups. All480 snapshots were sealed before any new development query readout.

## Exploratory state/action description

A retrospective summary of existing saved decision traces found that ENTRY100 deployment states have, on average,6 of24 feature dimensions outside their training ranges; the mean maximum absolute normalized coordinate is4.418 and nearest-training RMS distance1.126. These range/distance summaries are not calibrated OOD tests, and the repeated ENTRY100 rows are not independent observations. They motivate examining training-state coverage, not a claim that extrapolation uniquely caused the failure.

At ENTRY100, seed601 RL has mean policy entropy1.027 versus CE1.430, and alpha=.75 probability mass.795 versus.749. Lower entropy co-occurs with worse deployment but does not prove a causal entropy effect. The target-matched control remains nearly identical to RL. No new inference, training, query or feature extraction was performed for these descriptive summaries.

## Verification and full physical costs

All26 jobs and the detached coordinator exited0; no failures, process ended and lock released. Root and per-job optimizer ledger keys agree exactly:48,008 native updates and4,097 actor updates,52,105 paired attempted/successful calls. The one data ridge solve has its own paired ledger and residual check. All1,936 probe extractions and720 query calls have matching attempt/success records;2,880 query-image evaluations were consumed. All240 reward formulas and policy means,480 action/probability records, complete matrix keys and the sealing barrier were checked. Private raw ledgers, states and checkpoints remain onNAS.

Elapsed wall time was64.82minutes. Cumulative campaign totals:523,365 native updates excluding8,000 common-source updates (531,365 including),173,265 actor updates,51 data plus4 synthetic linear solves. This round also counted202,656 labeled and199,776 unlabeled item loads,399,552 student +99,888 EMA +199,776 memory image forwards. These repeated accesses are not new unique examples or new annotations.

## Next bounded question and limits

The next proposed intervention is broader training-state coverage using only existing A_fit/U_adapt and Q_train roles, while keeping the12-action dictionary, stable probes, policy architecture/objectives and stronger matched-target control. Freeze the exact additional transformations, entries, collection behavior, matched fitting and costs before execution. Preserve old results; do not tune on individual development maximizing actions or change reward/readout mid-run.

The same four development contexts and query images have been repeatedly observed in this campaign. The new random streams vary future optimization, not source checkpoints, patients or domains. Retention is simulated within the first domain. Dense training reward and hinge-penalty deployment utility differ. All results remain development evidence; no patient-independent benefit, clinical improvement or GRPO efficacy is claimed. Hidden U labels, sealed test and real later domains remain unused.
