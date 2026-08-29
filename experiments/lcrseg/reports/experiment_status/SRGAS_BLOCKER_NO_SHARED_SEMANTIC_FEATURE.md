# BLOCKER

## Scope

Fundus / SR-GAS V0.1 / pre-implementation model-path gate / frozen `UNet2D`

## Symptom

`HARD_STOP_NO_SHARED_SEMANTIC_FEATURE`: the existing relation path consumes the
64-channel `dec3` tensor, while the segmentation classifier consumes the
16-channel `dec1` tensor. No existing shared tensor has the classifier input
channel dimension required by the registered A5 broadcast formula.

## Reproduction

Instantiate `UNet2D(3, 3)`, attach forward hooks to `dec3`, `dec2`, `dec1`,
`segmentation_head`, and `projection_head`, then forward `[1,3,384,384]` and
differentiate the segmentation and relation outputs separately.

## Evidence

- `lcrseg/models/unet.py:55-76`
- `lcrseg/models/projection_head.py:16-27`
- `lcrseg/methods/lcrseg_v0_2a.py:649-660`
- `reports/experiment_status/SRGAS_MODEL_PATH_AUDIT.json`

## Confirmed facts

- `dec3=[1,64,96,96]` is the exact relation-projection input.
- `dec1=[1,16,384,384]` is the exact segmentation-classifier input.
- Classifier weight input channels are 16.
- `L_rel` reaches `dec3` but not `dec2` or `dec1`.
- `L_sup` reaches all three decoder tensors.

## Unknowns

The preregistration does not define a channel mapping from `dec3` sensitivity
to the 16-channel classifier weight, nor authorize moving the relation head to
`dec1`.

## Prohibited workaround

Do not choose another feature layer, alter the frozen relation projection,
insert a learned/fixed channel mapping, or reinterpret the 64-channel vector as
16 channels without a protocol amendment.

## Proposed next action

Obtain a protocol amendment that uniquely specifies either (a) a common
classifier/relation feature tensor with matching channel dimension, or (b) a
deterministic, frozen 64-to-16 sensitivity mapping and the corresponding new
acceptance tests. Then version the amended method before implementation.
