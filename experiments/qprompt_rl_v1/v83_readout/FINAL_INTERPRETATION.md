# V8.3 and V8.3-R interpretation

1. Original V8.3 gate: **STOP_DENSE_REWARD_NO_PRACTICAL_GAIN**; original screen reproduced without changing thresholds.
2–3. Cross-stream conditional value over global AND time controls: **True** under the prespecified engineering gate.
4. Both LOCO seeds exceed both fold-trained controls: **False**.
5–6. GRPO increment and warm-start total-cost benefit: NOT_TESTED by this readout; do not infer them from dense supervision.
7. New/old tradeoffs and absolute forgetting are listed below for all methods; utility×100 is not a Dice change.
8. Cross-stream and LOCO findings locate evidence gaps within fixed D1 contexts. Reward timing and deployment distribution shift remain hypotheses, not established causes.

Continuation status: **STOP_V83_PRIMARY_GATE_FAILED**.

| Method | New Dice | Old Dice | Utility | Utility ×100 | Mean absolute forgetting (pp) |
|---|---:|---:|---:|---:|---:|
| NATIVE | 0.734741107 | 0.812819198 | -0.073255628 | -7.325563 | 5.026801 |
| FIXED_BEST | 0.737623854 | 0.810925372 | -0.072268197 | -7.226820 | 5.216184 |
| UNIFORM_ACTION | 0.737752672 | 0.812890666 | -0.070331457 | -7.033146 | 5.019655 |
| PRE_FROZEN_601 | 0.736709660 | 0.812735751 | -0.071903184 | -7.190318 | 5.035146 |
| PRE_FROZEN_602 | 0.736709660 | 0.812735751 | -0.071903184 | -7.190318 | 5.035146 |

Absolute forgetting is memory-reference Dice minus endpoint Dice (signed; no deadband). The reward penalty applies a separate 0.005 deadband.

Cross-stream deltas (raw reward):

| Direction | State − uniform | State − global | State − time | Global − uniform |
|---|---:|---:|---:|---:|
| 1->2 | 0.001890505 | 0.000797835 | 0.000635653 | 0.001092670 |
| 2->1 | 0.001932307 | 0.000825752 | 0.000661926 | 0.001106555 |
| both | 0.001911406 | 0.000811794 | 0.000648789 | 0.001099612 |

Rank agreement: mean Spearman 0.8979166666666667; best-action agreement 0.8125. Constant rank vectors are missing, not zero.

Evidence limits: repeated D1 development, image-level roles, shared query images across random streams, shared image/model sources across LOCO folds. The 288 rows, 16 states and two actor seeds are not independent patients or independent student repetitions.
Label budget: 16 M_fit + 8 A_fit + 4 Q_train_old + 4 Q_train_new + 4 Q_dev_old + 4 Q_dev_new = 40 labeled D1 images; support 2/8 is only the adaptation subset. U_memory/U_adapt contain 80 images each, with hidden labels unused.
Any dense-supervision PASS supports only complete-action reward-supervised development diagnostics, not GRPO efficacy or independent generalization.

## What the readout establishes

The conditional-action engineering gate passes: both cross directions improve over global and time-fixed selection, and their bidirectional means are +0.000811794 and +0.000648789 raw reward. This is training-stream reproducibility on the same query images, not independent generalization.

The mapping gate fails: LOCO seed 601/602 improve over fold-trained global controls by only +0.000030226/+0.000031090, while losing to fold-trained time controls by −0.000250827/−0.000249963. The complete-table actors are worse than even the global control by about 0.00070. Their within-time action-probability L1 variation is only 0.0018–0.0032 on average. These observations support investigating the learning recipe and state conditioning; they do not establish that these states are intrinsically unlearnable.

Both formal seeds sampled exactly the same two actions in each development context under the matched private deployment streams, explaining their identical endpoint scores. Their probabilities are not identical, and the two checkpoints are retained. There is no missing second seed and no seed selection.

Compared with uniform, both dense policies change new Dice by −0.104301 pp and old Dice by −0.015491 pp, while utility changes by −0.001571727. Absolute forgetting remains about 5.04 pp relative to the frozen memory reference. This is not a better learning/retention tradeoff. V8.4 was not run, so neither GRPO extra-stage benefit nor warm-start benefit with or without preparation cost has been tested.

## Additional exploratory observation and next hypothesis

After all registered analyses were read, a zero-update analytic check evaluated exact reproduction of the registered soft-preference targets. Its in-table expected reward is 0.001810089, below the time-fixed policy by 0.000114863 and slightly below the global policy. This is a same-table diagnostic, not an independent test or a universal performance upper bound. The soft target itself can dilute a reproducible action-selection advantage; simply reducing target-fit loss does not necessarily improve the requested reward comparison.

The next economical investigation is a separately registered CPU-only attribution of target sharpness and input conditioning. Any new target or preprocessing belongs to that new experiment. None is inserted into V8.3, its readout, or the disallowed V8.4 path. Deployment-state shift, longer-horizon reward mismatch, clinical utility, patient independence, and real subsequent-domain generalization remain unverified.

## Accounting and engineering completeness

V8.3 used 34,008 new student updates and 133 new actor updates. The cumulative campaign totals are 224,853 student and 837 actor updates, plus 8,000 historical common-source student updates disclosed separately. The 29,600 dense-information preparation cost is verified as 800 native ENTRY200 updates plus 28,800 candidate updates; shared reuse does not make this supervision free.

V8.3-R used 0 student updates and 1,025 CPU actor updates (1,024 LOCO + 1 qualification). Its first invocation stopped before any optimizer call because a guard compared NAS mtimes with the training host clock. The repaired invocation used a common filesystem clock and completed. Both the failed script/log and successful outputs remain on NAS; PATCH_LOG.md records the evidence. No source campaign was rerun. Physical forward-call counts, kernel GPU time and peak VRAM were not instrumented in historical V8.3; per-job process durations are provided but are not substitutes for those quantities.
