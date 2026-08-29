# SPARC-Seg V0.1 model-path audit

**Status:** `SPARC_MODEL_PATH_AUDIT_PASSED`  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`

## Actual paths

- `dec3`: the existing `UNet2D.dec3` output, shape `[B,64,H/4,W/4]`.
- `dec1`: the existing `UNet2D.dec1` output and direct segmentation-classifier input, shape `[B,16,H,W]`.
- `relation_features`: the unchanged existing `projection_head(dec3)`, shape `[B,128,H/4,W/4]`.
- `logits`: the unchanged existing `segmentation_head(dec1)`, shape `[B,3,H,W]`.
- Decoder features in `SegModelOutput` are the same tensor storages observed by forward hooks; no second encoder/decoder pass, adapter, projection, 64-to-16 mapping, or cross-layer comparison exists.

## Before/after one-batch golden

- logits max abs: `0.0`
- relation max abs: `0.0`
- dec3 max abs: `0.0`
- dec1 max abs: `0.0`

## Old/current compatibility

All six registered consecutive-site seed pairs were loaded strictly. Same-name decoder tensors, relation features, and logits have identical shapes and dtypes; every exposed decoder tensor shares storage with the corresponding original forward tensor.

## Gate

Failed checks: `[]`.

No optimizer step or hidden-GT access occurred. Historical checkpoints and data were read-only.
