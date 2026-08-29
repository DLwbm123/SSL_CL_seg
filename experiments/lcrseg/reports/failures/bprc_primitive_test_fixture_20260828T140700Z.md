# BPRC primitive test-fixture failure

**Classification:** engineering-only test fixture error  
**Optimizer steps:** `0`  
**Checkpoint/data mutation:** `none`

The first audit-primitive test run reported `24 passed, 1 failed`. The failing test raised `TypeError: must be real number, not list` while constructing its synthetic one-pixel score tensor, before calling the BPRC loss. The implementation tests that reached the loss passed.

Correction: replace the malformed nested Python list with `torch.tensor([3.0, 1.0, 0.0]).reshape(1, 3, 1, 1)`. No formula, threshold, data view, or source checkpoint changed.
