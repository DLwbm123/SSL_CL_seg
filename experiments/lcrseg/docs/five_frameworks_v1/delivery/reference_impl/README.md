# Reference implementation — exact scope

本目录是独立数学算子与synthetic测试，供Codex参考和接入测试，不是完整训练器。

Implemented:
- masked forward KL / masked equal-image-class JML1;
- complementary source collection;
- convolution-channel readout proxy with effective C @ F_prev;
- constant-memory StageSubspaceAdapter and exact linear stage merge;
- source-isolated R gradient path and explicit anisotropic custom backward;
- current-L gradient thin-SVD initializer;
- empirical equal-cardinality sliced W2;
- detached finite-step coordinate repair with injected shape loss;
- current-label prototype estimates and teacher-only PAS-inspired selection;
- fail-closed review workflow check.

NOT implemented/NOT verified:
- actual KI parent binding, network bridge and real optimizer/EMA loop;
- full F1–F5 integrated runners, data loaders, restore/controller/evaluator;
- the CWMI backend and official D-Convexity backend;
- production CUDA/AMP/DDP and native segmentation-model deployment parity;
- class-conditioned sampler around SWD and current-stage prototype EMA;
- real-data studies or research results.

Repair tests inject a toy penalty solely to verify gradient and state semantics.
Do not call those tests a D-Convexity implementation test.

`gradient_scale_identity` intentionally defines a custom update: its backward is
not the derivative of its identity forward. Validate the stated rule, not an
ordinary finite-difference gradient check of identity.

Requirements used by this code: Python >=3.10, PyTorch, pytest. The handoff test
record states the actual local environment; it is not a requirement to upgrade
the project's existing environment. Codex must test the real target environment.

```bash
PYTHONPATH=. python -m pytest -q
```
