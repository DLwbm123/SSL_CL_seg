# V3A2 final interpretation

All 12 cells and all registered methods completed. Primary evaluation is sealed deterministic transfer on seeds262/263, h5. Softmax and h1 are secondary only.

Primary A EFFECT−BASE: (1.481754588894546e-05, 1e-06). Primary B EFFECT−FINE: (6.399938138201833e-05, 9.517389116808772e-06). Each pair is (mean quality difference, numerical repeat band).
Cells above their net-value band: 5/8. Backbone net values: {'UNET_QUERY_128': 5.9418147429823875e-05, 'DINOV2_VITS14_QUERY': 6.858061533421278e-05}. Frozen proposal criterion: False.

## Feature variation and predictability
FEATURE_VARIATION gives the actual variation of all 40 entry-state dimensions and within-scene repeat variation. FEATURE_REWARD_CORRELATIONS gives descriptive h1/h5 correlations without independent-sample significance claims. SCALER_VARIATION uses frozen training means/stds; constant training columns remain zero at transfer.

## Matched learner and transfer
DEVELOPMENT_OOF and MAIN_CONTRASTS separate local grouped OOF from sealed cross-prefix transfer. EFFECT−BASE is the state-information comparison within this protocol; changes from V3A also include common pool, repetition and execution-rule changes.
Transfer EFFECT−NC_OPT: (6.61129888612777e-05, 1.941734808497131e-06); EFFECT−SHUFFLE: (-4.784342794058224e-05, 1.2427626643329859e-05). Context utility cannot be inferred solely from improving a harmful baseline.

## Non-RL controls and numerical resolution
EFFECT−EFFECT_RIDGE: (-1.6258942196145654e-05, 4.868779797106981e-06); EFFECT−EFFECT_RULE: (-4.7732037880147495e-05, 5.055786459706724e-05). These controls remain mandatory; do not claim RL-specific advantage if they explain the gain.
REPEAT_ERROR and HORIZON_TRANSFER retain raw values and worst-scene spans. Results within a band are numerically inconclusive, not set to zero or excluded. Three repeats are not a confidence interval.

## Returning to FINE and future work
TRANSFER_POLICY_VALUES reports FINE frequency, intervention rate and per-action utility. A return to FINE without positive net value is not intervention success. No closed-loop pilot, new medical endpoint, retained student update, original KI substitution or automatic V3B was executed.

## Limits
U development and transfer pools are disjoint including continuation images. Optimization seeds were used previously; L cohorts and feedback identities are shared. This is not patient-independent confirmation, external validation, or an unbiased long-term return estimate. Feedback quality is not Dice percentage points. Publication requires separate current permission.

## Eight evidence-based answers

1. **Entry-state variation:** in each of all 12 cells, 12/16 original dimensions and 16/24 added dimensions have nonzero across-scene variance. Thus the new continuous information varies, while neither all original dimensions nor all added dimensions vary. Raw and standardized variances are retained in the corresponding tables.
2. **Relation to h1/h5:** across transfer cells, added dimensions and the two non-FINE action targets, median absolute descriptive correlation is 0.357 at h1 and 0.304 at h5 (maxima 0.953 and 0.794). These are exploratory summaries of many correlated columns, not predictive validation or significance tests; instantaneous fit-gradient alignment does not guarantee delayed audit value.
3. **Matched learner:** transfer h5 EFFECT−BASE is +1.48175e-5, exceeding its 1e-6 repeat band. This supports a positive aggregate state-information contrast within this frozen protocol, not an across-the-board improvement.
4. **OOF versus transfer:** grouped development OOF EFFECT−FINE is −9.92036e-5 versus BASE −9.15795e-5 (EFFECT−BASE −7.62412e-6); transfer EFFECT−FINE is +6.39994e-5. The evidence does not describe an OOF success that consistently transfers; the two evaluations differ in prefixes and U pools and should both be retained.
5. **Simple controls:** transfer EFFECT_RIDGE value is +8.02583e-5 and EFFECT_RULE +1.11731e-4, versus EFFECT_POLICY +6.39994e-5. The policy is below ridge by 1.62589e-5, beyond the paired 4.86878e-6 band. The rule gap is within its 5.05579e-5 band. No RL-specific advantage is established; the shuffled-reward policy also exceeds EFFECT_POLICY by 4.78434e-5, beyond its 1.24276e-5 band.
6. **Numerical resolution:** aggregate A and B both exceed their registered repeat bands, but only 5/8 individual cells have EFFECT−FINE above their cell bands. All eight remain in the averages. Three repeats quantify execution variability only, not confidence intervals, statistical significance or patient independence.
7. **Returning to FINE:** EFFECT chooses FINE on 63/128 transfer scenes (49.21875%); it intervenes on 65. Across all scene/repeat evaluations, negative interventions occupy 19.0104%. NC_OPT returns to FINE on 50% of scenes and its aggregate value is −2.11361e-6. Returning to FINE is attainable but is not by itself positive intervention value. EFFECT's positive aggregate value does not remove harmful individual interventions.
8. **Pilot condition:** NOT MET. The registered requirement is at least 6/8 cells above the B band; observed 5/8. The required positive contrast against EFFECT_SHUFFLE also fails. Positive A/B aggregate means and positive means in both backbones do not override those failures. No closed-loop pilot is authorized or launched.

## Execution and delivery

The run finished on 2026-09-30 at approximately 00:15:45 Asia/Shanghai, under the original deadline. All 576 feature extractions, 2,304 VJPs, 1,728 virtual previews, 1,728 five-step branches and 60 neural fits completed. The physical ledger contains 8,640 real disposable student updates and 15,360 controller updates, plus the separately reported synthetic qualification costs. There were no failed or replayed calls. No medical checkpoint or retained update was produced.

Completion audit verifies immutable prefix and schedule bindings, original optimizer groups and scheduler provenance, seal hashes, and prediction-before-transfer-update ordering. Branch receipts verify Adam 2001–2005, local scheduler 1–5, teacher EMA updates and exact full-root restoration. Feature access restrictions are application-level guards; they are not an adversarial operating-system sandbox. Qualification parity results are synthetic tests, while actual CUDA execution variability is measured by the three repeats.

The terminal FINAL receipt is authoritative: the supervisor stage file retains its last RUNNING/report text after normal exit, and both recorded parent and last child processes have exited. This stale stage label is not an ongoing optimizer.

Source and anonymized summaries are prepared locally. Private manifests, case-level payloads, feature vectors, gradients, raw logs and checkpoints are excluded. GitHub publication awaits separate permission for this round.
