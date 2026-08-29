# CRISP-Seg V0.1 model-path revalidation

**Status:** `CRISP_MODEL_PATH_REVALIDATION_PASSED`  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`

CRISP reuses the SPARC-exposed `dec3` and `dec1` tensors and does not modify U-Net. All six registered seed-transition pairs and both old/current roles strict-loaded against their frozen checkpoint state dictionaries.

- Parameter count: `507731` (constant across all probes)
- State-dict key SHA256: `965afdc2958d24f83f961e85b60a5697f91a0abe4963ee894997fdd289cb7271` (constant across all probes)
- Logits max abs versus pre-SPARC golden: `0.0`
- Relation-feature max abs: `0.0`
- dec3 max abs: `0.0`
- dec1 max abs: `0.0`
- Failed checks: `[]`
