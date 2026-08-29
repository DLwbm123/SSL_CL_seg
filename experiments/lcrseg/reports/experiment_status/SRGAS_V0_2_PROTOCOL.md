# SR-GAS V0.2 Frozen Protocol

Status: `SRGAS_V0_2_PREREGISTERED_IMPLEMENTATION`

SR-GAS V0.2 preserves the correct V0.1a hard stop and the original worst-trajectory safety threshold. The only method changes are:

1. L2/L3/L4 consume sensitivity from the previous successful optimizer step.
2. L1/L2/L3/L4 linearly raise classifier-noise amplitude from zero to `sqrt(0.1)` during the first fixed 20% of successful site steps.
3. L1-L4 use a stateless shared raw standard-normal stream keyed by protocol seed, split seed, site, successful site step, and classifier-weight shape.

The class-space R2C definition, 0.5/0.5 source weights, inverse-minmax scale, cosine classifier, R0 learner, relation KD, pseudo-labeling, anchor lifecycle, network architecture, and full-amplitude variance remain unchanged.

Registered seed-0 pilot variants:

| Variant | Sensitivity | Timing | Warm-start |
|---|---|---|---|
| L0 | none | none | none |
| L1 | isotropic ones | none | 20% |
| L2 | clean total | previous successful step | 20% |
| L3 | supervised | previous successful step | 20% |
| L4 | supervised + R2C | previous successful step | 20% |
| D1 | supervised + R2C | same step | 20% |
| D2 | supervised + R2C | previous successful step | none |

D1/D2 are diagnostic-only and cannot replace L4. D2 intentionally retains full-amplitude noise at its first incremental step because it is the no-warm ablation; the zero first-step perturbation contract applies to the registered warm-start variants.

The fixed L4-vs-L0 worst-drop threshold is `0.015` independently on REFUGE and RIM-ONE-r3. Endpoint recovery cannot override that threshold.
