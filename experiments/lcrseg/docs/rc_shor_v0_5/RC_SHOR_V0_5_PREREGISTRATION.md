# RC-SHOR V0.5 preregistration

RC-SHOR V0.5 is a new grouped outer-OOF experiment. It does not change SHOR V0.3.1 (`FAIL_SELECTIVE_OVERRIDE_STABILITY`, H5 false) or SHOR V0.4 (`PASS_FIXED_POLICY_TEST_EFFECTIVENESS`). It starts at `6e42a04c4ea0547aeb89d430f96b551294cc3aaf` on `codex/shor-v0-5-rc-stability` and must produce exactly one pre-evaluation commit and one report commit.

## Isolation and population

The evaluator population is the 49 seed-case rows (37 unique Fundus patients/cases) whose seed-specific role is `train_labeled` and whose case and patient are never validation or test in any seed. The same patient/case across seeds is one group. Five outer folds are assigned by sorting `sha256("rc-shor-v0.5-fold\0" + patient_id)` and round-robin assignment. An outer fold's labels and domain are unavailable until its candidates are sealed. Only the other four folds may construct utility targets or select that fold's candidate. V0.4 `formal_03` is forbidden until the final V0.5 status exists and is not used by this experiment.

## Frozen inference and features

The nine B0 student snapshots, three seed-0 EMA descriptor snapshots, 102-dimensional raw descriptors, three frozen stage-2 ridge states, original SHOR thresholds, original five bootstrap-threshold rows, preprocessing, deterministic inference, tie handling and inactive-NaN rule are immutable. Frozen forwards may materialize descriptors and probabilities; segmentation training, optimizers and parameter updates remain zero.

Each historical expert has a 141-dimensional feature vector: 102 raw descriptor values; three ridge probabilities; its log alpha contrast to stage 2; top1-top2 alpha margin; ridge entropy; six statistics for each of three expert probabilities (mean predictive entropy, mean foreground probability, hard rim area, hard cup area, total 4-connected foreground-component count, normalized hard foreground boundary length); and, for each of three expert pairs, mean Jensen-Shannon divergence, hard-mask disagreement, rim-area disagreement, cup-area disagreement and boundary disagreement. No GT or true domain is a feature or RC-SHOR argument.

## Utility model and selection

For historical expert `h`, utility is its case foreground Dice minus stage-2 foreground Dice. Independent weighted ridge regressors use lambdas `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`. Grouped five-fold cross-fitting selects minimum OOF MSE, breaking ties toward larger lambda. One-sided conformal correction is the maximum 0.90 higher quantile of `prediction - utility` over nonempty seed/domain groups in inner-training cross-fitted residuals.

One hundred deterministic patient-cluster bootstrap fits are used. Patients are resampled within historical/current dataset domains; duplicate seed rows inherit one patient multiplicity. Inactive rows retain NaN predictions/LCBs and are never filled. A replicate is feasible only with at least 15 active unique patients, finite active values and every required nonempty residual group. OOD means the maximum absolute base-standardized feature exceeds 8. C6 chooses the lowest-index historical expert on an exact median-LCB tie and overrides only when median LCB exceeds epsilon, its 100-replicate vote fraction meets rho, at least 90 replicates are feasible, base support is at least 15 and the row is finite and not OOD.

The fixed candidate grid is rho `{0.70,0.80,0.90}` by epsilon `{0,0.005,0.010}`. In each outer fold, candidates are ordered by: all inner stability/safety gates; larger historical-gain p10; smaller current-drop p90; smaller maximum seed-domain-drop p90; larger rho; larger epsilon; larger selected lambda (simpler ridge); candidate ID. This total order also gives a deterministic fail-closed primary candidate if no candidate clears the first criterion. Outer results never alter selection.

## Controls and decisions

C0 is stage 2; C1 frozen ridge hard; C2 frozen ridge soft; C3 exact frozen S3; C4 requires rho consensus across the five original frozen SHOR bootstrap-threshold routes repeated deterministically to 100 votes; C5 uses the non-bootstrap utility LCB and selected epsilon; C6 is full RC-SHOR; C7 is the true-domain snapshot oracle; C8 is the per-case best frozen expert. C7/C8 are evaluator-only.

The JSON preregistration fixes all metric definitions, bootstrap seeds, thresholds, gates, status precedence, artifacts and hard stops. After the single formal attempt, no threshold, grid, feature, fold, candidate or stopping rule may change. A post-hoc V0.4 comparison is permitted only after immutable status and cannot change it.
