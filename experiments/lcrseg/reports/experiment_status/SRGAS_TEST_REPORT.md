# SR-GAS V0.1a Test Report

Status: `SRGAS_V0_1A_TESTS_PASSED`

The mandatory V0.1 and V0.1a contract tests passed (`50/50`), and the complete historical repository suite passed without assertion changes (`148/148`). The run used Python 3.10.6, PyTorch 2.2.1+cu121, CUDA runtime 12.1, and pytest 8.3.5.

Verified boundaries:

- cosine classifier is bias-free and normalizes features and weights in float32;
- only the classifier weight is sampled, without in-place master-weight mutation;
- R2C uses a detached historical relation target in the frozen Fundus class order;
- R2C has no gradient path to the old model, historical anchors, or projection head;
- R2C is sensitivity-only and is absent from the final optimizer objective;
- no channel mapping or U-Net architecture change exists;
- empty-mask and first-site behavior reduce A5 exactly to A4;
- spatial shuffle preserves target marginals, breaks alignment, and resumes from checkpoint exactly;
- an actual A1-parent-to-A5 incremental runner step produced valid R2C pixels and no old-model gradient;
- all earlier LCR-Seg regression tests remain green.

Commands:

```text
PYTHONPATH=.:tests python -m pytest -q tests/test_srgas_v0_1.py tests/test_srgas_v0_1a_r2c.py
PYTHONPATH=.:tests python -m pytest -q tests
```

Next registered gate: `A1_TWO_CASE_OVERFIT`.
