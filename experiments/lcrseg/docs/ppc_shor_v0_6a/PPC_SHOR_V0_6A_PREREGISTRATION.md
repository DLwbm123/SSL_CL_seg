# PPC-SHOR V0.6A preregistration

PPC-SHOR (Partially Pooled Calibrated Selective Historical Override) is a development-only
feasibility study. It asks whether cross-seed partial pooling can stabilize the frozen SHOR
score while preserving historical value and current-domain safety. Every existing Fundus row
is development-consumed; no result can be described as external confirmation.

The fixed calibration population contains 990 frozen stage-2 train-memory OOF rows from 575
patients. The segmentation-value population contains all 198 seed-case rows whose own seed
role is `train_labeled`, representing 177 patients; val/test-role segmentation GT is forbidden.
Cross-seed patient duplicates stay in one of five deterministic outer folds and have total
analysis weight one. All rows of an outer patient are excluded from inner calibration.

For each historical expert, deterministic weighted PAV fits one pooled and seed-local isotonic
curve from raw SHOR log-alpha contrast. Kappa is 10, 30, 100, or infinity; tau is 0.90, 0.95,
or 0.98; rho is fixed at 0.80. The primary stability engine is a 200-replicate patient Bayesian
bootstrap within dataset domain. Delete-one-patient jackknife and ordinary clustered bootstrap
are sensitivities. Consensus uses the number of finite feasible predictions, never an assumed
200. A zero-route precision is undefined (`null`) with explicit numerator and denominator.

Before any outer segmentation GT access, all 11 design gates in the JSON registration must pass
and all candidate routes, calibration models, and bootstrap weights must be sealed and verified.
Any design failure yields `BLOCKED_DESIGN_DEGENERATE_BEFORE_GT`; no conservative fallback may
continue. Candidate selection uses inner domain labels only. The code freeze is pushed and its
remote SHA verified before the sole authorized development outer-OOF execution.

V0.3.1, V0.4, and V0.5 remain immutable with their existing statuses. V0.4 `formal_03` has zero
authorized reads. There is no main merge or external test in this protocol.
