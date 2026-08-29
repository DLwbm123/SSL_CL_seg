# SPARC-Seg V0.1 source audit

**Status:** `SPARC_SOURCE_AUDIT_PASSED`  
**Generated:** 2026-08-28T23:25:00+08:00  
**Scope:** source semantics and adaptation boundary only; no method implementation or optimizer step

## JASCL: paper/source-derived evidence

- Official repository: `https://github.com/prinshul/JASCL`.
- Audited commit: `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53` (required exact commit matched: `True`).
- Prototype source: `methods/utils.py::get_prototype` and `norm_mean`.
- Source normalizes each pixel feature, averages pixels within each case, then averages case prototypes. It does not explicitly normalize the final cross-case mean and does not enforce SPARC's 32-cell minimum.
- PAS source uses softmax confidence and class-wise cosine similarity with strict `> 0.7` / `> 0.7` comparisons.
- Student and EMA-teacher label maps are filtered independently. The source computes MSE directly between the two filtered integer-valued maps; no explicit Boolean student/teacher-mask intersection is constructed.
- The teacher update is EMA with `alpha=0.99`, and PAS is invoked periodically after prototype refresh.

### Paper/source distinction and SPARC adaptation

The paper-level contribution describes prototype-assisted pseudo-label validation and a mean-teacher consistency path. The exact per-case reduction, final-normalization omission, filtered-label-map MSE, and absence of an explicit intersection are source-level details. SPARC does not silently copy those implementation details: it freezes thresholds at 0.7/0.7, uses current plus frozen-previous validators, has no EMA teacher, and retains the R0 hard-CE pseudo target.

## STAR: paper/source-derived evidence

- Official repository: `https://github.com/jinpeng0528/STAR-TPAMI`.
- Audited repository HEAD: `6c9203aa0c91e9a2d4e40664c97754ae02226675`.
- The old-class region in the verifiable VOC implementation is `(current label == background) AND (old-model prediction > background)`. The old prediction uses thresholded old logits: pixels with no old logit above 0.5 are reset to background.
- `PKDLoss` compares `features[5]`, which is the ASPP/classifier-input tensor `x_pl` appended last by `DeepLabV3.forward(..., ret_intermediate=True)`.
- It uses elementwise MSE, bilinear-resizes the region mask, and divides the masked sum by `mask_sum * channels`.
- No explicit source flag implementing an all-feature PKD ablation was found in the audited checkout; SPARC's registered S5 all-valid-spatial control is therefore a protocol control, not a claimed STAR-source reproduction.
- SPARC adopts only the targeted-maintaining idea. Its frozen contract is dual-stable foreground, channel-normalized cosine distance at same-name layers; it does not copy STAR's region rule, bilinear soft mask, layer, or MSE formula.

## LAG: conceptual provenance only

LAG motivates stable/sample-specific semantic information as a concept. SPARC does not implement LAG's channel-wise split, spatial decoupling, asymmetric contrastive module, NSC, or LRP.

## Frozen SPARC source boundary

- `thresholds = 0.7 / 0.7`
- current plus frozen-previous semantic validators
- no EMA teacher in proposed SPARC
- R0 hard-CE pseudo target remains unchanged
- no cross-site prototype memory or replay
- no STAR formula is transplanted
- no LAG module is implemented

## Integrity checks

- `jascl_confidence_default_0_7`: `PASS`
- `jascl_cross_case_mean`: `PASS`
- `jascl_ema_alpha_0_99`: `PASS`
- `jascl_exact_required_commit`: `PASS`
- `jascl_filtered_map_consistency_mse`: `PASS`
- `jascl_origin_official`: `PASS`
- `jascl_per_pixel_normalize_then_mean`: `PASS`
- `jascl_periodic_pas_schedule`: `PASS`
- `jascl_prototype_function`: `PASS`
- `jascl_similarity_default_0_7`: `PASS`
- `jascl_strict_joint_filter`: `PASS`
- `jascl_teacher_filtered_independently`: `PASS`
- `star_bilinear_region_resize`: `PASS`
- `star_elementwise_mse`: `PASS`
- `star_feature_5_is_classifier_input`: `PASS`
- `star_feature_index_5`: `PASS`
- `star_mask_channel_reduction`: `PASS`
- `star_old_class_region_rule`: `PASS`
- `star_old_logit_threshold`: `PASS`
- `star_origin_official`: `PASS`

## Reference file SHA-256

| File | SHA-256 |
|---|---|
| `JASCL methods/utils.py` | `59701733130bc003ca62beb0b3cb399bd6e0a401315d8edfdf71db428fbf09aa` |
| `JASCL methods/trainer.py` | `2f42c1a5ccdc950060d75d610b44223df96076164e3fb2a9ff4fc08dd83f495b` |
| `JASCL run.py` | `76913839f890b097c671b2399347132c347eb008346a4739c72624a92ae8ebc6` |
| `STAR models/loss.py` | `a346d8c6c3680d65cf347bf06e0800d87c6ce4f3e14476a647f82539bc889e21` |
| `STAR models/model.py` | `14f0a45028ea65a3c9612686fbcf67e5c3d55e43d992713da6cbd54aaf951889` |
| `STAR trainer/trainer_voc.py` | `fa7ad3620d3c18d227418a3f0b7d42b5658e359a620df12fa8e1a69543d5ef37` |
