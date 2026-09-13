# Deviations, limitations and pending binding

Implemented and tested: all five independent framework loss/gradient paths; five common baselines; shared mixing and current-stage EMA/prototypes; source-preserving input construction; CPU structural backends; stage sealing; exact synthetic checkpoint resume; fault-tail accounting; finite synthetic controller; full C/D DAG; selection and metric helpers; fail-closed CLI. Existing execution code and historical terminal artifacts were not modified.

Explicit adaptations needing external review:

- CWMI uses the author's locked complex N=2, K=4 steerable pyramid, foreground classes, no author CE, and unnormalized real-embedded complex conditional covariance. Prediction covariance ridge is the author's 5e-4. A **target conditional covariance ridge of 5e-4** plus FP64 symmetric solve/Cholesky is added for constant/empty-class stability. This is disclosed, not called exact author-default equivalence. Value/gradient parity is tested with target ridge zero on nonsingular random inputs; the stabilized integrated path separately passes finite/gradient tests on actual pyramid outputs and constant fields. Full valid images are used; ignored images use fixed nonoverlapping 128x128 all-valid crops.
- D-Convexity follows locked MIT `C2nd_loss` including its epsilon/softplus/gradient-magnitude behavior. The implementation exposes a spatial map for conservative 3x3 backward-stencil geometry erosion and applies it to disc union and cup. It is neither TV nor exact CGPM. Differences from the paper's displayed ReLU are the author code's softplus relaxation; this source choice is fixed.
- F4 trust is a per-image scalar feature RMS radius. Inner repair uses the original teacher probability and frozen original PAS mask; three detached gradient steps do not guarantee a lower shape loss or better Dice.
- JML uses image/foreground-class reduction, no smooth additive numerator, explicit double-empty zero and teacher detach. Author parity uses norm=1, alpha=beta=gamma=1, smooth=0, mIoUI=1 on nonempty support. It is not the entire JDT family.
- Synthetic Adam and the toy foreground Dice are **synthetic fixtures**, not replacements for unknown real parent optimizer or native CE+Dice. The bridge must delegate to those verified real native definitions. Synthetic scale calibration uses the usual median, averaging the middle two ratios for eight values.
- R2 implements device-local CWMI factories and separate per-device caches; exact CPU author-pyramid output/gradient parity passes. CUDA and formal mixed-precision qualification remain NOT_RUN; implementation is not accelerator qualification.
- Future E ablations are registered and plan-only, deliberately rejected by the executable arm registry. Only explicitly supported options (e.g. lambda_JML/SWD/shape zero, kappa zero) can be used in synthetic comparisons; full E contrasts need their separate review.

Unfinished because original parent identity is unavailable:

1. Real ParentBridge binding, native geometry/readout and stochastic-head behavior, A/B ordering/ranks, hard versus soft constraint callbacks, optimizer/scheduler/scaler, original data capabilities and source provenance. Existing recovery audits say `PARENT_NOT_LOCATED_IN_ACCESSIBLE_SCOPE`. F_CONV, SVD and old weight-memory implementations were not substituted.
2. Real source/target budgets, feature widths and effective-rank alias deduplication, legal source reuse, full seed-collision freeze. A bounded source-config search found no 161–164 seed declarations; this is not a claim that all unavailable historical runs were checked.
3. Real reader binding, original native stage execution, isolated native evaluator, stochastic-layer replay and CUDA qualification remain pending. R2 adds NativeParentBridge delegation, current-domain data adapter schema, finite metadata readiness and actual approval preflight. The real runner registry stays empty; editing a receipt cannot create an executable adapter.

Minimum missing identity clue: **one original KI launch command or one original run_id**. Historical score tables are not requested. The package is ready for module-level external review and cannot receive a truthful full-real-integration PASS yet.

R01–R06 are addressed in [REVIEW_RESPONSE_R1.md](REVIEW_RESPONSE_R1.md); current exact evidence is TEST_REPORT_R2.json. The prior 104-test report remains under history and at b909a913.
