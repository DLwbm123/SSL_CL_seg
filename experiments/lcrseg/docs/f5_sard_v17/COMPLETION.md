# V17 complete: higher mean, original dual gate still fails

All8target stages sealed, all4trajectory EXIT0, controller/process EXIT0, completion audit PASS. The original20seen-domain evaluations ran after the complete seal. Execution fe00f79c0200d5991628b2445eae8f8840710003. Formal21200/source0, no retry or partial patient readout.

| Metric | V17 (%) | minus V13 (pp) | minus V15 (pp) | minus matched F5 (pp) | minus historical B2 (pp) |
|---|---:|---:|---:|---:|---:|
| Final | 68.5113 | +0.6855 | +0.2428 | +3.1004 | +0.6102 |
| Old | 67.3405 | +1.0133 | +0.5698 | +4.1717 | +1.0995 |
| Incoming | 70.8527 | +0.0298 | -0.4113 | +0.9577 | -0.3685 |

Mean Final versus B0 is +3.7072pp; versus V12 +1.1823pp and V10 +1.5560pp. V13 is the matched common base supported by disabled native parity. B0/B2 remain historical comparisons. Full paired results, stage scores and every positive/negative class cell are retained in delivery/RESULTS.json; V12 is an explicitly derived background comparison from its previously delivered aggregate.

| Seed/order | Final (%) | Old (%) | Incoming (%) |
|---|---:|---:|---:|
| 163/1 | 67.1855 | 65.4769 | 70.6027 |
| 163/2 | 67.2645 | 65.6511 | 70.4914 |
| 164/1 | 70.3749 | 68.8650 | 73.3947 |
| 164/2 | 69.2201 | 69.3692 | 68.9220 |

## Original gates and requested tradeoff

Both required gates fail. Against B2, the mean clears +0.5pp, but seed164 average-order Final is -0.2581pp (seed163 +1.4784pp); order2 Incoming is -1.6280pp, below the -0.5pp floor; seed164/order1 RIM rim is -5.9444pp, below the -5pp cell floor. Against F5, seed163/order1 REFUGE cup -5.3803pp and RIM cup -5.8219pp breach that same cell floor. No threshold or checkpoint choice is changed. Passing the separate V13 comparison does not replace the original F5/B2 success criterion.

Seed164/order1 RIM rim is58.0020%, versus V13 58.3813% (-0.3793pp) and V15 60.7438% (-2.7418pp). Its stage1-to-stage2 change is71.9068%→58.0020% (-13.9047pp). The requested rim problem remains. Seed164 average Incoming is -0.0444pp versusV13 and -1.2024pp versusV15, although +0.5753pp versusB2; order-level failure must not be hidden by this average.

Seed163/order2 Drishti cup improves to57.8314%, +2.1586pp versusV13, +11.9909pp versusV15 and +5.6020pp versusB2. Its own stage1-to-stage2 change is71.2405%→57.8314% (-13.4091pp). Spatial weighting preserves more of this cup result than V15's independent class cone, but does not fix all negative cells or establish a spatial cause of forgetting.

## Activation and cost

All16960active calls used nonuniform spatial weights, with no zero-score class fallback. All50880present-class reductions occurred. The unchanged aggregate entry halfspace projected8956/16960calls; A/B/R U gradients were nonzero on all16960active calls. These receipts confirm execution, not reference correctness or preservation of old domains. The score could emphasize an incorrect feature structure; current U does not establish old-domain coverage, and raw-gradient constraints do not guarantee Adam's finite update behavior.

Formal training:89040student full forwards,50888teacher full forwards including16960entry anchors,16960reference VJPs,64fixed R diagnostic VJPs. SARD computed5001707520feature-pixel scores and1125384192000neighbor vector distances; this is substantial additional arithmetic despite no extra model forward or VJP. Controller wall time3955.344seconds (65.92minutes); peak reserved CUDA2302672896bytes per trajectory. Individual runtime/memory and full operation counts are public receipts.

Campaign actual formal403473, including historical673rollback/lost calls; copied sealed5300 not counted twice. This round qualification32CPU/10generatedCUDA/2currentL, cumulative960/237/48, source0. Prior failures, GUA's zero formal advancement and five historical missing diagnostic rows remain. All receipts in delivery/ are anonymous aggregates; patient-level records, private configuration, checkpoints and raw logs remain onNAS.

## Next independent authorized candidate

After this complete public delivery is verified, proceed to REDistill on the common V13 base, without stacking SARD. The next question is reference robustness, not further tuning these spatial weights. First verify the original power-divergence direction, fixed parameter, zero-probability semantics and wrong-teacher gradients, then freeze and qualify a separate bounded adaptation. No additional formal trial, parameter search or old-data access is authorized by this report itself; use the existing four-direction queue and its per-candidate limits.

These are repeatedly reused development-patient results, not independent generalization, SOTA or clinical benefit.
