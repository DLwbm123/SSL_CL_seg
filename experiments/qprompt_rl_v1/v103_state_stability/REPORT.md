# V103 fixed-probe state stability

**A fixed common probe improves cross-stream state-to-action prediction in this training table; averaging four probes is not uniformly better than one. Condition-heldout results remain mixed, and no deployment or independent-source gain is established.** Completed 2026-10-08T14:17:23.959362+00:00.

## Frozen intervention and qualification

Primary data are the existing V10132training states and384training returns. Keep the exact24Dextractor, weights, EMA, memory, checkpoint step and learning rate. Replace the stream-specific batch inputs with four predetermined probes: provider seed168, cursors0/1/2/3, global RNG seeds861030/861031/861032/861033. Restore provider seed, cursor, all model/optimizer state and RNG after each call. Compare ORIGINAL, FIXED_SINGLE(probe0) and FIXED_MEAN4(float64 componentwise average). No probe-count, seed, model or ridge-lambda search.

Before any new feature extraction, all32original state vectors replayed exactly. All160extractions asserted full native snapshot/RNG preservation, with provider seed checked explicitly; empty optimizer capability prohibited updates. The existing V102 predictor helper is reused. ORIGINAL reproduces all320V102V101 heldout evaluation rows exactly. Thirty trainfold-only ridge fits use lambda1 and row-centered raw returns;1NN uses the same train-only normalization. Every one of960individual evaluation rows is retained.

## State variation and prediction

| Representation | Step100 cross-stream RMS | Step200 cross-stream RMS |
|---|---:|---:|
| ORIGINAL | 0.028902610 | 0.067372340 |
| FIXED_SINGLE | 0.000000000 | 0.003549381 |
| FIXED_MEAN4 | 0.000000000 | 0.001573610 |

All eight step100pairs become exactly equal under either fixed-probe representation. Their shared student checkpoints and common probe now produce common features; the original batch/RNG dependence caused their previous differences. At step200the student trajectories differ, so residual state differences remain. RMS uses raw heterogeneous feature units; it is descriptive and is not a calibrated measure of predictive information or independent reward noise.

Values below are heldout **training dense reward minus train-global/time**, not Dice percentage points or development utility. Global and time controls coincide in these folds.

| Representation | Predictor | Test stream1 | Test stream2 | Condition-LOCO mean | LOCO positive / zero / negative |
|---|---|---:|---:|---:|---|
| ORIGINAL | STATE_RIDGE | -0.000022716 | +0.000349430 | +0.000125433 | 2 / 3 / 3 |
| ORIGINAL | STATE_1NN | +0.000625882 | -0.000282900 | +0.000460875 | 6 / 0 / 2 |
| FIXED_SINGLE | STATE_RIDGE | +0.000440016 | +0.000174944 | +0.000287165 | 3 / 3 / 2 |
| FIXED_SINGLE | STATE_1NN | +0.000853946 | +0.000852056 | +0.000219761 | 6 / 1 / 1 |
| FIXED_MEAN4 | STATE_RIDGE | +0.000426497 | +0.000331777 | +0.000414674 | 3 / 3 / 2 |
| FIXED_MEAN4 | STATE_1NN | +0.000853946 | +0.000852056 | +0.000359862 | 6 / 1 / 1 |

Both fixed representations make ridge and1NN positive against both controls in both stream directions. However, fixed-mean ridge is slightly worse than ORIGINAL on test stream2, and slightly worse than fixed-single on test stream1. Averaging improves the ridge condition-LOCO mean relative to both alternatives but still leaves two negative conditions. Fixed-mean1NN has the same stream results as fixed-single; its condition-LOCO mean is lower than ORIGINAL. Therefore this is evidence for stabilizing batch-dependent state extraction, not proof that four-batch averaging is universally preferable.

The stream-swap result has a strong condition-matching advantage: the training fold includes the other stream of the same condition/time, and step100fixed states are now identical.1NN achieves exactly the earlier V102condition-matched top-action transfer values. This must not be sold as unseen-condition generalization. Condition-LOCO removes that exact condition from training, but the same source and query roles still underlie all conditions; independent patient/source validation remains NA. No statistical population claim is made from eight overlapping conditions.

## Execution, recovery and cost

The initial launch omitted the established CUBLAS_WORKSPACE_CONFIG=:4096:8 and exited1 during model construction, before state extraction or image forwards. Its original source, failure, exit, log and two successful synthetic ridge solve pairs are preserved onNAS. Recovery explicitly restored that environment setting, reused the completed synthetic qualification and resumed once from untouched frozen checkpoints into a new output subdirectory. No budget reset or rerun of consumed solves occurred. The recovery finished with captured exit0 in about64seconds; live EXEC_RUN/start identity, neutral ps/nvidia command lines and the held lock were checked, followed by process exit and lock release.

Cumulative V103physical work:32original-state replays +128new probe extractions =160;30data ridge solves +2synthetic solves;30nearest-neighbor reference sets. Each extraction and solve has paired attempted/success entries. There was one startup failure, zero failed extraction or solve, zero native/actor optimizer update, and zero training/development reward-query evaluation.

Inference is not free: dataset hooks counted640labeled and640unlabeled item loads; model hooks counted1,280student +320EMA +640memory image forwards. The run read8existing A_fit support images and15existing U_adapt images in total; repeated loads are included. No new annotation case was introduced, no hidden U label entered extraction, and Q_train/Q_dev images were not opened. Existing Q_train reward numbers were reused. Private image identities and feature arrays remain onNAS.

Campaign totals remain475,357nativeupdates excluding8,000common-source updates (483,357including), and87,245actorupdates. V102andV103together additionally consumed50data linear solves and4synthetic solves, recorded separately from optimizer steps. The320row ORIGINAL replay is a reuse check, not new scientific replication.

## Interpretation and next bounded question

This intervention supports batch dependence as a practical weakness in the current state representation. It does not identify the unique cause of V101development failure, establish RL incremental gain, or establish an independent-domain effect. The common probe can improve condition recognition while still missing transferable action-relevant information.

A next training-only test can compare matched CE/RL learners using stable probe features under strict fold isolation, with any warm-start fitted exclusively inside the training fold. Fix seeds, objective, initial policy, scaler, update budget and simple predictor controls before fitting. Do not reuse a full-table historical actor as a heldout initialization. This would test whether the learned policy objective generalizes beyond the simple-predictor diagnostic before another native reward campaign. New deployment or independent-source experiments need their own concrete protocol and data boundary assessment; no such experiment has started here.

The method's key isolation, A-only adaptation and history-free parameter memory are unchanged. All negative conditions, all representations and the startup failure remain in the release. Public artifacts contain only source, plans, aggregate/condition-level metrics and accounting; frozen weights, state vectors, identities and raw images/labels remain private.
