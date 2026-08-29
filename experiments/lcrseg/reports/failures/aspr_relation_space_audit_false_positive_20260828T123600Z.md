# ASPR relation-space audit failure bundle

**Classification:** `AUDITOR_FALSE_POSITIVE`  
**Original reported status:** `HARD_STOP_ASPR_RELATION_SPACE`  
**Optimizer steps:** `0`

The first audit preserved at
`reports/experiment_status/ASPR_RELATION_SPACE_AUDIT.{md,json}` failed only the
check `classifier_relation_anchor_axes_agree`.

## Root cause

The auditor incorrectly required three descriptive strings in frozen
`class_semantics.json` to be literally equal. The frozen semantics intentionally
state:

- segmentation classifier axis = `training class_id order`;
- anchor-bank axis = `training class_id order`;
- relation-distribution axis = `anchor bank class_id order`.

The last string defines the relation axis by reference to the anchor-bank axis.
It is therefore semantically identical without being textually identical. The
frozen class order and raw-to-training mapping were already exact in the failed
audit.

## Evidence retained from the first audit

- All nine required R0 site checkpoints were present and loaded strictly.
- Every checkpoint had relation dimension 128 and valid normalized class anchors.
- Site 0 had no historical anchor; sites 1 and 2 had valid historical anchors.
- All six consecutive-site old/current pairs produced finite normalized
  `[1, 128, 64, 64]` relation features for the same probe.
- Relation temperature was 0.1 throughout.
- Manifest and split hashes matched every checkpoint.
- Physical GPU 5 was `GPU-2052f3b4-88f8-4be9-d43c-5068fafb02a5`.

## Correction policy

The original failed audit is not overwritten or deleted. The check is corrected
to validate the explicit two-edge semantic bridge, and a new immutable audit
artifact is generated under a distinct filename. No dataset, manifest, split,
checkpoint, model, or prior report is changed.

Original SHA-256:

- JSON: `1f4f4a6315d570fcf42bab07857423f9513f505d096f1a272ec8f9f60e2fe662`
- Markdown: `464cd6dc8f28e55ea2aee40a6948662b6914c16a1645641965127f1803dcd581`
