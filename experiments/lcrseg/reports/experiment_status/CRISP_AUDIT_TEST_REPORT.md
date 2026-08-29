# CRISP-Seg V0.1 audit-only test report

**Status:** `CRISP_AUDIT_TESTS_PASSED`  
**Executed environment:** `/home/jiangsuiyang/anaconda3/envs/py38/bin/python` on `zmic44`  
**Result:** `13 passed in 1.35s`

Executed after the zero-vector subgradient repair:

```text
python -m py_compile \
  lcrseg/representation/channel_roles.py \
  lcrseg/representation/style_probe.py \
  lcrseg/losses/channel_role_consistency.py \
  scripts/audit_crisp_sources.py \
  scripts/revalidate_crisp_model_paths.py \
  scripts/audit_crisp_style_probe.py \
  scripts/audit_crisp_feasibility.py \
  scripts/compile_crisp_feasibility.py

python -m pytest -q \
  tests/test_crisp_channel_role_primitives.py \
  tests/test_crisp_style_probe.py \
  tests/test_weak_strong_geometry_alignment.py
```

Coverage includes exact `(F*grad)^2`, case-equal aggregation, continuous alpha/beta and zero evidence, deterministic split halves, C4/C5 controls, role-state round trip, same-geometry/no-cutout style views, global-RNG isolation, IFC previous-branch stop-gradient, PFC two-branch gradients, and finite zero-vector subgradients.

The conditional full CRISP method test suite and historical training regression were not entered because feasibility ended at the preregistered hard stop and no method/config was registered.
