# Failure and mechanism attribution

All statements below describe this fixed development cohort and sealed predictions. No known bad-case identity was available to the feature, fitting or deployment APIs.

- CURRENT: historical-route source mismatches 0 seed rows / 0 patients; current-domain subset 0 rows / 0 patients; all-domain macro delta < -0.10 in 0 rows / 0 patients.
- RIDGE_HARD_FROZEN: historical-route source mismatches 4 seed rows / 4 patients; current-domain subset 3 rows / 3 patients; all-domain macro delta < -0.10 in 3 rows / 3 patients.
- SHOR_FROZEN: historical-route source mismatches 2 seed rows / 2 patients; current-domain subset 1 rows / 1 patients; all-domain macro delta < -0.10 in 1 rows / 1 patients.
- PPC_FROZEN: historical-route source mismatches 2 seed rows / 2 patients; current-domain subset 1 rows / 1 patients; all-domain macro delta < -0.10 in 1 rows / 1 patients.
- SHOR_CONFIDENCE_VETO: historical-route source mismatches 2 seed rows / 2 patients; current-domain subset 1 rows / 1 patients; all-domain macro delta < -0.10 in 1 rows / 1 patients.
- SHOR_UV_CF: historical-route source mismatches 1 seed rows / 1 patients; current-domain subset 0 rows / 0 patients; all-domain macro delta < -0.10 in 0 rows / 0 patients.
- SHOR_UV_ROUTED_ONLY: historical-route source mismatches 2 seed rows / 2 patients; current-domain subset 1 rows / 1 patients; all-domain macro delta < -0.10 in 1 rows / 1 patients.
- SHOR_UV_GAIN_ONLY: historical-route source mismatches 2 seed rows / 2 patients; current-domain subset 1 rows / 1 patients; all-domain macro delta < -0.10 in 1 rows / 1 patients.

The full private failure table includes every class-harm or source-mismatch row in every policy, across both historical and current domains. Identity checks use actual frozen route/domain records; they are not inferred from 154/155. Source error does not itself imply segmentation harm.

Target composition (two counterfactual experts, not independent patients):
- REFUGE expert 0: 4/120 class-harm pairs; rim 1, cup 3; macro-beneficial 120; frozen SHOR selected 120.
- REFUGE expert 1: 56/120 class-harm pairs; rim 51, cup 39; macro-beneficial 76; frozen SHOR selected 0.
- RIM_ONE_r3 expert 0: 13/48 class-harm pairs; rim 6, cup 10; macro-beneficial 42; frozen SHOR selected 1.
- RIM_ONE_r3 expert 1: 5/48 class-harm pairs; rim 0, cup 5; macro-beneficial 48; frozen SHOR selected 36.
- Drishti_GS expert 0: 29/30 class-harm pairs; rim 29, cup 25; macro-beneficial 3; frozen SHOR selected 0.
- Drishti_GS expert 1: 30/30 class-harm pairs; rim 30, cup 26; macro-beneficial 2; frozen SHOR selected 1.

Actual incremental comparison: MIXED_TRADEOFF_OR_POSSIBLE_INCREMENTAL_VALUE. Consult ablation and paired tables rather than treating a better ablation as a main-method win.

Failed gates: VALUE:overall_gain, VALUE:historical_gain, BASELINE:overall_difference_from_SHOR, BASELINE:historical_difference_from_SHOR.

The H head predicts a point value, not an upper confidence bound. Correcting a rare source mistake is distinct from reliably predicting its class loss. Small current-domain support and prior development exposure limit every safety/utility conclusion.


## Verified current-error identity and veto mechanism

Private identity equality, using actual seed/case/patient records, confirms that SHOR and PPC retain the same single current-domain seed-case error. CF vetoes it through H_hat > the selected inner harm_limit despite positive predicted macro gain. Routed-only and gain-only retain it. Only aggregate counts and equality are public; individual targets/predictions/identities remain in the NAS witness.

CF discards 24 macro-beneficial historical calls while removing the one macro-harmful current call. Six accepted historical rows still have class harm. The observed pattern supports a value/safety tradeoff and inadequate utility retention, not a claim that CF is useless or that class safety is established. Current-domain supervision includes 59/60 harmful counterfactual pairs; routed-only has only the one selected current pair. These paired observations are not independent patients and do not provide an external causal test.
