# ASPR-Seg V0.1 relation-space audit

**Status:** `HARD_STOP_ASPR_RELATION_SPACE`  
**Optimizer steps:** `0`  
**Workspace hash:** `014a9ce51c0bcea1ad9adbd0c4264f26a5e010cd8d8911cee4e256c1e1f4895e`

## Frozen contract

- Feature source: `UNet2D dec3 -> existing ProjectionHead(64, relation_dim)`.
- Relation dimension: `128`; relation grid: existing one-quarter resolution.
- Fundus class order: `0 background`, `1 optic_disc_rim`, `2 optic_cup`.
- Site-memory foreground IDs, if feasibility later passes: `[1, 2]`.
- Existing relation temperature: `0.1`.
- Existing valid mask: strict full-cell valid pooling on the relation grid.
- Existing class anchors remain class-semantic anchors and are not replaced.

## Checkpoint coverage

| Seed | Site | SHA-256 | Relation dim | Current anchors | Historical anchors |
|---:|---|---|---:|---|---|
| 0 | REFUGE | `609410775b56f073bddecf4a482667bdcc7be31deb29e71d9a74931303f197bb` | 128 | valid | not applicable |
| 0 | RIM_ONE_r3 | `1e7f0aad58c4e734d38899cc6be327494a4e762ab7889ec57c16c19cc33bf660` | 128 | valid | valid |
| 0 | Drishti_GS | `82aae753208cf94e832ae80a1f5336c4238dbd6ba7845ab7e2e4ce857d8d339d` | 128 | valid | valid |
| 1 | REFUGE | `c8a8c95e0fdddde88acdc42c3cbc62a3f20ac436a2cf00f76513ed29b410ab0b` | 128 | valid | not applicable |
| 1 | RIM_ONE_r3 | `b746eb3c6a2bb4aea731fdd0cb0b79bb09e231dda26962d162a329b19fe504cd` | 128 | valid | valid |
| 1 | Drishti_GS | `64c8b6c78d95c5ee196f40fd32797bb79674d14c8d578b3bb29511c32bae384f` | 128 | valid | valid |
| 2 | REFUGE | `7f2645fc6ccae039004616ef9f68c36990ffb4f0b33dd606286ac0edc82617fb` | 128 | valid | not applicable |
| 2 | RIM_ONE_r3 | `8c955e6f37e1013e08c1007aeb7ff5f59a566d26371e828ce8aeeb31b5cf1093` | 128 | valid | valid |
| 2 | Drishti_GS | `53c52e720a7e6a51bd4207f0fee46b3f03f1598751372ee23d7b38aa0b6db283` | 128 | valid | valid |

## Old/current pairing

All six consecutive-site model pairs were loaded strictly and probed with the same input tensor. Their logits and normalized relation-feature grids must match exactly in shape.

## Gate

Failed checks: `['classifier_relation_anchor_axes_agree']`.

No data, split, manifest, prior report, prior run, or checkpoint was modified. No optimizer step was executed.
