# LCR-Seg V0.2a freeze and V0.3 preregistration

**Date:** 2026-08-27  
**V0.2a status:** `FUNDUS_V0_2A_GATE_FAILED`  
**V0.3 status:** `PREREGISTERED_NOT_YET_RUN`

## Frozen V0.2a conclusion

- R1 is the single-factor positive signal.
- R2 and R3 are frozen negative results.
- Calibrated teacher rejection is not part of V0.3.
- V0.3 makes no method-formula change relative to V0.2a R1.
- Existing runs must not be deleted, renamed, or overwritten.

| Variant | Role | Final Dice | BWT | Incoming | Previous |
|---|---|---:|---:|---:|---:|
| R0 | formal legacy continuous + uniform relation | 0.6551054533 | -0.1184621104 | 0.7340801936 | 0.6759458886 |
| R1 | progressive admission + uniform relation | 0.6616491375 | -0.1162918645 | 0.7391770472 | 0.6826418677 |
| R2 | legacy continuous + teacher rejection | 0.6129432916 | -0.1942860025 | 0.7424672933 | 0.6181338590 |
| R3 | progressive admission + teacher rejection | 0.6328992030 | -0.1056452933 | 0.7033293985 | 0.6908223824 |
| U0 | auxiliary unit-all + uniform relation | 0.6309953259 | -0.1111381492 | 0.7050874253 | 0.6687593930 |

## Frozen V0.3 candidate

```text
Progressive Learnability-Guided Admission
+ Uniform Semantic Relation Consolidation
```

The schedule remains classwise, per-site, and linear from 0.40 to 0.80.
Teacher rejection, multi-agent, RIC, EWC/MAS/SI, gradient projection, third
teachers, and additional losses are excluded.

## Frozen input hashes

| Seed | Training manifest SHA-256 | Fundus split SHA-256 |
|---:|---|---|
| 0 | `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3` | `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88` |
| 1 | `d5d2913054bc96f13b2baec0f21109a7da92c1a2f5b07f0cde234b35bbfd92a9` | `87affde62045894a8ce89701137f254ed56ba1f00951041bd2f6282cccbb5727` |
| 2 | `78379dc43035259f41b0f598e0bda25a31e68b15600bb611758ccc61cd2a0727` | `af2f48281d8eb16d299871f12824a729d08cb3854b3753d69d42c0d842e34dd3` |

The frozen V0.2a R1 REFUGE site-end checkpoint SHA-256 is
`9bdadf34a5a32d936b14cfff3f4c9ffa2ee62c5f24142ca12b4a3b9815c46b32`.

## Registered execution order

1. P0 seed-0 site-1 exact bridge and P0 continuation.
2. Fundus R0/R1 seeds 1 and 2 in the preregistered counterbalanced order.
3. Internal multi-seed gate.
4. Conditional strong baselines.
5. Conditional P0 seeds 1 and 2 and frozen post-hoc analyses.
6. Conditional Prostate RUNMC to BMC pilot only after both Fundus gates pass.

