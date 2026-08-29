# CRISP-Seg V0.1 source audit

**Status:** `CRISP_SOURCE_AUDIT_PASSED`  
**Generated:** `2026-08-29T10:00:00+08:00`  
**Scope:** provenance and adaptation boundary; no CRISP method registration or optimizer step

## LAG source idea

- Official repository: `https://github.com/YBIO/LAG` at `3dbd3f8dcee06770aa9c4412f447147b2520c71f`.
- The paper/repository motivates semantic-invariant versus sample-specific representation roles and channel-wise/spatial decoupling.
- The audited source exposes `rho` as a CLI channel-allocation parameter, computes a channel count, and compares same-index old/current features.
- The audited source default is `rho=1.0`; CRISP C4's fixed 60/40 hard split is a preregistered contextual control, not claimed as the audited source default.
- CRISP does not copy LAG prototype matching, triplet/contrastive implementation, LRP/NSC, unknown-class handling, or a fixed split into the proposed C3 method.

## STAR source idea

- Official repository: `https://github.com/jinpeng0528/STAR-TPAMI` at `6c9203aa0c91e9a2d4e40664c97754ae02226675`.
- STAR provides evidence for partial rather than global feature stabilization: its audited PKD path applies a spatial region mask to a selected old/current feature tensor.
- CRISP does not copy STAR prototype replay, background repetition, pseudo-label region rule, selected layer, bilinear soft mask, or MSE formula.

## DC²T source idea

- Primary publication: `https://doi.org/10.1109/TMI.2024.3469528` (IEEE Transactions on Medical Imaging 44(2):903-914, 2025).
- The publication describes online semi-supervised representation disentanglement, content-inspired parameter consolidation, and style-induced consistency training.
- No official implementation was located in the bounded source search, so only publication-level claims are used. CRISP does not claim source-code reproduction.
- CRISP does not copy DC²T's dual encoder, VAE, FiLM, CPC parameter consolidation, SCT parameter perturbation, or reconstruction path.

## CRISP adaptation

CRISP's case-equal `(F*grad)^2` content score, centered normalized paired-view style score, continuous `alpha=Cn/(Cn+Sn)`, complementary `beta`, and dual IFC/PFC allocation are protocol-specific adaptations. CRISP is not a direct implementation of LAG, STAR, or DC²T.

## Integrity checks

- `crisp_not_direct_reproduction`: `PASS`
- `dc2t_primary_doi_recorded`: `PASS`
- `lag_channel_count_from_rho`: `PASS`
- `lag_channel_wise_decoupling_stated`: `PASS`
- `lag_contrastive_path_present`: `PASS`
- `lag_origin_official`: `PASS`
- `lag_rho_cli_present`: `PASS`
- `lag_rho_source_default_recorded`: `PASS`
- `lag_same_index_current_feature`: `PASS`
- `lag_same_index_old_feature`: `PASS`
- `lag_sample_specific_role_stated`: `PASS`
- `star_classifier_input_feature`: `PASS`
- `star_origin_official`: `PASS`
- `star_partial_region_mask`: `PASS`
- `star_pkd_training_path`: `PASS`
- `star_selected_feature_pair`: `PASS`

## Reference SHA256

| File | SHA256 |
|---|---|
| `LAG README.md` | `8a16d24aea54089ba61c4e9ca3b145961b70a94a24a793a1e2a4670b939911b0` |
| `LAG run.py` | `b497112755ffcd75be5cd9f302a746698c684bc1ef6bb57f363d8ae0acb0896b` |
| `LAG utils/contrastive_learning.py` | `a91112a7b6d92510e6a63e3b53615ed267f71a2e7351d1740cc3e6aecb8e65e4` |
| `STAR models/loss.py` | `a346d8c6c3680d65cf347bf06e0800d87c6ce4f3e14476a647f82539bc889e21` |
| `STAR models/model.py` | `14f0a45028ea65a3c9612686fbcf67e5c3d55e43d992713da6cbd54aaf951889` |
| `STAR trainer/trainer_voc.py` | `fa7ad3620d3c18d227418a3f0b7d42b5658e359a620df12fa8e1a69543d5ef37` |
