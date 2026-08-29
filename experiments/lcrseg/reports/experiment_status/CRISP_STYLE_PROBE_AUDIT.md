# CRISP-Seg V0.1 style-probe audit

**Status:** `CRISP_STYLE_PROBE_AUDIT_PASSED`  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`  
**Pseudo-label usage:** `none`

The paired role-probe views share one deterministic horizontal/vertical flip. The style view applies only the already registered contrast, brightness, and Gaussian-noise path. Cutout is disabled because it changes spatial support.

- Contract SHA256: `7d9c2f9a43b0a6c688de8c9543713bd78779231a32f7d4669d98ffd560399ec3`
- Frozen magnitudes: flip `0.5`, noise std `0.03`, brightness delta `0.1`, contrast delta `0.1`
- Fundus: three-channel intensity path; no new hue/saturation/color operator.
- Prostate MRI: the same existing operations on a single intensity channel.
- Case RNG key: `protocol_seed + site_id + case_id`; the global PyTorch RNG is unchanged.
- Failed checks: `[]`
