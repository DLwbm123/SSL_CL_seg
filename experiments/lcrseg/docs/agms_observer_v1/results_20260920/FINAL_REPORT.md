# AGMS_OBSERVER_V1 final results

Completed 2026-09-19 22:52:21 Asia/Shanghai. Both new models are VERIFIED; all 10 cost sessions closed PASS, with zero formal failures. DESCRIPTIVE_RECOVERY=true for both orders. The detached auxiliary-head intervention recovers approximately the B2 baseline; it does not establish an improvement over B2 or A1.

Execution commit: `f4a1088efe7ea570de0c64d67a1a9d7051ddaa83`. External AI review canonical digest: `b4775116c63e10e3fff9f070432d11059e43bccdd0503652522d8418ae44f9c0`. This results-only publication leaves the reviewed execution checkout and historical results unchanged.

## Frozen experiment and acceptance

Only the auxiliary-head feature inputs are detached relative to original A5; DS remains 0.25. Seed 163, two stage-2 nodes, reused original B2 stage-1 prefixes: O1 Drishti 2100 updates; O2 RIM 3200 updates. No new source, stage-1 or P0 execution. B2/A1/original A5/HALF are eight historical imports; two OBS0 endpoints are new.

Physical costs: formal 5300, L-only smoke 8, generated CUDA qualification 10 (including one prescribed failure-injection call), real total 5308. Diagnostics: 8 points / 32 additional VJPs; no additional backbone forwards. Preparation CPU remains 12 calls over 2 attempts, with historical 235 separate; no launch-time CPU suite rerun. Export coverage: 10 endpoints, 30 domain rows, 12 paired rows, 8 diagnostic points.

## Endpoint comparisons

Values below are absolute Dice differences, averaged over the two orders. A0 denotes historical B2. Positive Final/Old/Incoming is better; source and first-target results remain separately available in DOMAIN_METRICS.csv.

| Contrast | Final | Old | Incoming |
|---|---:|---:|---:|
| OBS0-A0 | -0.000058092 | -0.000148854 | +0.000123430 |
| OBS0-A1 | -0.000078403 | -0.000131926 | +0.000028642 |
| OBS0-A5_CONTROL | +0.018197398 | +0.027309073 | -0.000025954 |
| OBS0-HALF_CONTROL | +0.017291169 | +0.023206532 | +0.005460444 |

Recovery requires each order Final >= B2 - 0.001, Incoming >= B2 - 0.005, and each old-domain macro >= B2 - 0.002. Both orders pass, including each separate old domain; source improvement cannot conceal first-target loss. Relative to original A5, incoming changes have opposite signs (O1 -0.013096421, O2 +0.013044514), so their near-zero mean is not uniform equivalence.

## Mechanism observations

At all eight committed diagnostic points, DS gradient norms on main A/B and auxiliary upstream A/B are exactly zero while auxiliary-head DS gradients are nonzero. This verifies the intended detach at the measured points. Recovery after removing upstream DS is consistent with that gradient path contributing to the original A5 degradation in these fixed trajectories; it does not establish a general causal claim across seeds or full trajectories. H remains active on the main network.

### Same-state U selections

Ratios use cumulative counts over active-U steps, not averages of step ratios. Main/equal/risk are evaluated on the same teacher state. U carries no diagnostic labels.

| Order | Active-U steps | Main eligible candidates | Equal / candidates | Risk / candidates | Equal-risk XOR / candidates |
|---|---:|---:|---:|---:|---:|
| O1 | 1680 | 12162119 | 0.806719 | 0.821394 | 0.017299 |
| O2 | 2560 | 3133939 | 0.929787 | 0.933228 | 0.004246 |

### Conditional disagreement versus main entropy on training L

Top quartiles match class, boundary region, and coverage within each diagnostic point. The following pools the resulting selected-pixel counts across those strata and four points per order; these are dependent training-pixel observations, not independent patients or calibration samples.

| Order | Selected pixels per ranking | Main errors captured by conditional disagreement | Main errors captured by entropy |
|---|---:|---:|---:|
| O1 | 51366 | 2910 | 11097 |
| O2 | 29735 | 3736 | 4743 |

Conditional disagreement captures fewer main-head errors than entropy in the pooled matched-coverage comparison in both orders. Individual strata are mixed; this comparison does not demonstrate incremental predictive value beyond entropy, and does not test a combined predictor. Parent/conditional Brier, class-balanced errors, regional counts, KL decomposition and all gradient points are preserved in the JSON exports. No conditional reweighting or online feedback was enabled.

## Limits, provenance and stop

Single development seed and two orders, not an independent-seed replication or significance test. Historical checkpoints share prefixes with new nodes; DeltaForget = -DeltaOld is the same common-prefix observation. Historical observer diagnostics remain NOT_MEASURED_HISTORICAL. Training-L analyses are not independent calibration. No claims of independent patients, original KI reproduction, or SOTA.

The operator-only preflight receipt initially failed to serialize immutable nested metadata after validation; the receipt serializer was corrected outside production code before any optimizer or real-payload reads. Original error evidence is retained privately. This was not a training retry; reviewed production code was unchanged.

Status: COMPLETE_OBSERVER_AWAITING_SCIENTIFIC_REVIEW. No automatic follow-up, extra seed, coefficient, route change or monitoring. Public files contain aggregate metrics and reproducibility metadata only; weights, raw logs, private paths and patient-level material remain excluded.

Inputs: PUBLIC_RESULTS.json; verification: COMPLETION_PROOF.json, COST_AND_COMPLETION.json, CUDA_QUALIFICATION.json, SMOKE.json; full endpoints: FINAL_METRICS.csv, DOMAIN_METRICS.csv; comparisons: PAIRED_COMPARISONS.csv; observer diagnostics: GRADIENT_DIAGNOSTICS.json, COVERAGE_AND_RISK.json.
