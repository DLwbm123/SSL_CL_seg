# SR-GAS V0.1a class-space bridge audit

**Status:** `SRGAS_CLASS_SPACE_BRIDGE_AUDIT_PASSED`  
**Class semantics SHA-256:** `5c52655356b11831820433035dad0adfe919219a4da2a9f70d2b18d784010200`

The audited class order is `0=background`, `1=optic_disc_rim`,
`2=optic_cup`. It is supported by the frozen raw-label mapping, model config,
formal Fundus experiment config, anchor-bank class indexing, and relation-field
class axis.

The runtime probe constructed `q_old_relation` only from a frozen previous
`UNet2D` and a non-parametric historical `AnchorBank`. Current clean logits
were downsampled with bilinear interpolation and `align_corners=False` to the
relation grid. The valid mask used the existing strict relation-grid contract,
required finite old targets, and required all historical class anchors.

The R2C proxy produced a finite nonzero gradient with shape
`[3, 16, 1, 1]` directly on the segmentation
classifier weight. It produced no gradient for the old model, historical
anchors, or current projection head. No channel mapping, hidden GT,
compatibility, or teacher rejection was used.

| Check | Result |
|---|---|
| `frozen_label_map_confirmed` | PASS |
| `fundus_class_order_exact` | PASS |
| `raw_mapping_values_cover_three_classes` | PASS |
| `model_config_class_count_is_three` | PASS |
| `experiment_config_class_count_is_three` | PASS |
| `experiment_dataset_is_fundus` | PASS |
| `no_channel_mapping_declared` | PASS |
| `architecture_unchanged` | PASS |
| `segmentation_and_relation_class_count_match` | PASS |
| `old_relation_probability_normalized` | PASS |
| `old_relation_target_detached` | PASS |
| `old_relation_from_frozen_previous_model` | PASS |
| `historical_anchors_are_nonparametric` | PASS |
| `historical_anchor_all_classes_available` | PASS |
| `current_logits_downsample_shape_exact` | PASS |
| `valid_mask_uses_strict_relation_grid_contract` | PASS |
| `r2c_valid_count_positive` | PASS |
| `r2c_loss_finite` | PASS |
| `r2c_classifier_gradient_shape_exact` | PASS |
| `r2c_classifier_gradient_finite` | PASS |
| `r2c_classifier_gradient_nonzero` | PASS |
| `r2c_projection_head_gradient_absent` | PASS |
| `r2c_old_model_gradient_absent` | PASS |
| `r2c_historical_anchor_gradient_absent` | PASS |
