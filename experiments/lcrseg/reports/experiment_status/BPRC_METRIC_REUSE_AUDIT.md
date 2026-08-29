# BPRC-Seg V0.1 exact TARC metric reuse audit

**Status:** `BPRC_METRIC_REUSE_AUDIT_PASSED`  
**Optimizer steps:** `0`

| Module | Function | Function SHA-256 | Semantics |
|---|---|---|---|
| `scripts.audit_tarc_relation_fidelity` | `_margin` | `8d6194382a82225faa97f00188905e53f7f512464e1821eecc15412c5ac49f57` | top1-minus-top2 probability margin |
| `scripts.audit_tarc_relation_fidelity` | `_previous_fidelity` | `4a89785523dde98ccaec825e5e3f20dc7e771b7288cce6a6c0f30ccf2f17a34c` | relation KL, top1 agreement, and classwise margin agreement |
| `scripts.audit_tarc_relation_fidelity` | `_current_safety` | `f7c3a995ccd38eec88208b73986ca232f495d9e2c6c5305b0240f626bce7397a` | current-site relation accuracy, margin, entropy, and finite checks |
| `scripts.audit_tarc_virtual_step` | `_supervised_r0_loss` | `447b7324d580c317039a28dbad2b93f6bf0eda5857f90e00453423843b0aa558` | exact frozen R0 supervised validation loss |
| `scripts.audit_tarc_virtual_step` | `_baseline_loss` | `cb2877728307d50011bfb702b70720803e559ec3aaea2bd11d3278c4a4f76b72` | fixed-batch baseline validation loss |
| `scripts.audit_tarc_virtual_step` | `_functional_val_loss` | `86f133c9cee2a6fd4b250664f0fd2f776be358d7cf9c2a9a89a6729e0a9a3088` | stateless functional-view validation loss |

The TARC source modules are frozen at their recorded SHA-256 values. BPRC feasibility must import these exact functions and may only add pairwise/class-balanced diagnostics that TARC did not define.
