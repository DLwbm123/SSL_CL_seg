# TARC-Seg V0.1 relation-space audit

**Status:** `TARC_RELATION_SPACE_AUDIT_PASSED`  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`

## Frozen relation contract

- Relation source: `UNet2D dec3 -> existing ProjectionHead(64, relation_dim)`.
- Feature dimension: `128`; grid: `64 x 64` for a `256 x 256` input.
- Class order: `0 background`, `1 optic_disc_rim`, `2 optic_cup`.
- Temperature: `0.1`; valid mask: existing strict full-cell relation mask.
- Site-0 has current anchors; later frozen checkpoints carry current and historical all-class anchors.
- TARC transports all classes, including background; no foreground-only fallback is permitted.
- Transport fitting is restricted to current-site `train_labeled` cases with at least 32 relation pixels per case/class.

## Visible current-site evidence

| Seed | Site | train_labeled cases |
|---:|---|---:|
| 0 | REFUGE | 40 |
| 0 | RIM_ONE_r3 | 16 |
| 0 | Drishti_GS | 10 |
| 1 | REFUGE | 40 |
| 1 | RIM_ONE_r3 | 16 |
| 1 | Drishti_GS | 10 |
| 2 | REFUGE | 40 |
| 2 | RIM_ONE_r3 | 16 |
| 2 | Drishti_GS | 10 |

## Gate

Failed checks: `[]`.

All nine site checkpoints and all six consecutive old/current model pairs were loaded strictly. No historical artifact or frozen data file was modified.
