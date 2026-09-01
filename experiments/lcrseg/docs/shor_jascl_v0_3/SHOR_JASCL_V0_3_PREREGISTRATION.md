# SHOR-JASCL V0.3 preregistration

## Identity and purpose

Registration ID: `SHOR_JASCL_V0_3_SELECTIVE_HISTORICAL_OVERRIDE`.

This is a new, zero-model-forward validation feasibility protocol addressing the current-domain contamination observed in the closed PRES-DSR-SF V0.2.1 ridge-soft candidate. It is not a V0.2.1 retry and does not modify descriptors, experts, ridge grids, temperature grids, or the soft-fusion control.

The protocol branches from `c854bd28b1a69ce001646201a824b8bb75141c67`. Closure commit `9feee43c5e34c427356ceaaafa6f691dd14186a3` binds the prior result `FAIL_SOFT_EXPERT_FUSION_VALUE` and the frozen private bundle: 183 files, 4,386,018,614 bytes, content SHA256 `05c9008ad4496ccbdc51df6103638024d49fae4b3b4cdc2a9f829c5f3ab165bb`.

## Frozen input and execution boundary

Only the sealed V0.2.1 descriptor, memory, router metadata, validation ordering, expert-probability caches, manifests, and guards may be read. Previous candidate segmentation metrics are not primary evidence. The source bundle remains read-only.

The run must construct no model and perform no model forward, autograd, backward, optimizer step, router optimizer step, parameter-gradient write, checkpoint tensor load beyond a hash audit, training, test construction, or test-GT read. The only fitting is closed-form ridge on train-only memory descriptors using the unchanged lambda and temperature grids.

No private array, validation GT, threshold fitting, candidate prediction, or formal result may be accessed or generated until this preregistration and a separate execution authorization are both published and remotely verified.

## Train-only threshold selection

At every seed and stage independently, five-fold train-memory OOF ridge probabilities are reconstructed. For current expert `t` and each historical expert `d < t`:

`top_i = argmax_j alpha_i,j`, with exact ties resolved to the lowest domain index.

`score_i,d = log(alpha_i,d + 1e-12) - log(alpha_i,t + 1e-12)`.

A row is accepted for historical override `d` iff `top_i == d` and `score_i,d >= tau_t,d`. Candidate thresholds are all unique finite OOF scores among rows with `top_i == d`, plus positive infinity.

For each threshold, compute accepted count, precision for true domain `d`, recall for true domain `d`, current-domain false-override rate, and false-override rate for every other domain. Feasibility requires precision at least 0.98, current-domain false-override at most 0.02, accepted count at least 15, and historical recall at least 0.35. Select highest recall, then highest precision, then highest threshold. No validation data enters selection. Missing any required seed/stage/historical-domain threshold yields `FAIL_SELECTIVE_OVERRIDE_CALIBRATION`.

## Policies and prediction

- `S0_SHARED`: current-stage expert for all cases.
- `S1_RIDGE_HARD`: frozen ordinary ridge top-1 hard route.
- `S2_RIDGE_SOFT`: frozen original soft fusion, retained only as the failure control.
- `S3_SHOR`: primary selective historical override.
- `S4_ORACLE`: fixed true-domain snapshot upper bound.

For stage `t`, let `d_star = argmax alpha`. If `d_star < t` and `score_d_star >= tau_t,d_star`, S3 routes to `d_star`; otherwise it routes to current expert `t`. Stage 1 falls back to expert 1 and stage 2 to expert 2. Threshold equality is accepted. There is no probability/logit averaging and no fractional expert weight. S1, S2, and S4 cannot rescue S3.

## Sealing, evaluation, and attribution

Before any validation domain or segmentation-GT read, seal all train OOF rows, thresholds, validation ridge alphas, top-1 routes and margins, historical/current alpha mass, SHOR score inputs, policy one-hot/alpha rows, candidate predictions, case ordering, and content hashes.

After the seal, the evaluator may read validation domain ID and segmentation GT. It reports the exact worst seed/domain; top-1 correct versus incorrect; soft-versus-current, soft-versus-hard, and per-case current-domain regret; historical alpha mass; and correct-route versus misrouted current cases. Attribution is descriptive and cannot change thresholds or candidates.

Five fixed train-memory bootstraps independently resample within each seed/stage/domain, refit unchanged ridge grids, reconstruct OOF predictions, and reselect thresholds. Bootstrap thresholds may not borrow the formal threshold. Cached validation descriptors and expert probabilities are reused with zero model forward.

## Frozen gates

- H1 calibration: every required unit has finite precision at least 0.98, current false-override at most 0.02, accepted count at least 15, and historical recall at least 0.35.
- H2 current safety, stage-2 S3 versus S0: current-domain foreground-Dice drop at most 0.010, every current foreground-class drop at most 0.015, and maximum seed-domain drop at most 0.020.
- H3 value, stage-2 S3 versus S0/S4: three-domain gain at least 0.100, historical-domain average gain at least 0.150, oracle gap at most 0.060, positive three-domain gain in all three seeds, and positive REFUGE and RIM_ONE_r3 mean gains.
- H4 repair of soft failure, S3 versus S2: current-domain drop reduction at least 0.020, maximum seed-domain drop reduction at least 0.020, shared-gain loss at most 0.060, and historical-gain loss at most 0.080.
- H5 stability: bootstrap shared-gain p10 at least 0.080, historical-gain p10 at least 0.120, current-domain-drop p90 at most 0.015, maximum seed-domain-drop p90 at most 0.025, every historical domain feasible in at least four of five replicates, and all values finite.
- H6 isolation: zero new model forwards/autograd/backward/optimizers/gradient writes, private input unchanged, no segmentation GT in threshold building, evaluator-only validation GT, zero test-GT reads, and complete output/archive hashes.

Only H1–H6 all true yields `PASS_SHOR_JASCL_VALIDATION_FEASIBILITY`. Scientific failure precedence is calibration, current safety, value, then stability. Engineering blockers take precedence over scientific adjudication.

## Attempt and hard stop

Subject to a separate authorization, exactly one create-only validation `formal_01` is permitted. Completion requires a durable server-local child exit receipt, private archive audit, sanitized publication, and remote verification.

After the report, stop for independent review. No test evaluation, second SHOR attempt, threshold-rule modification, validation refit, C0 regeneration, snapshot training, LoRA, adapter, Prostate, MnMS, sweep, or main merge is authorized. A scientific failure permanently stops the image-only domain-agnostic snapshot-routing line.
