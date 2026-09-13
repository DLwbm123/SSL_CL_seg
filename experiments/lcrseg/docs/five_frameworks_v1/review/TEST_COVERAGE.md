# Actual integration coverage and limits

The executable test list and per-case outcomes are recorded in TEST_REPORT.json. The user-supplied reference test count is not reused.

| Contract | Actual synthetic evidence |
|---|---|
| Shared L/U | Source-selection identity, distinct L patients, U source deduplication, ten-arm two-stage call graphs, no-U read counters, stateless augmentation/SWD namespaces |
| Losses | KL/soft-CE gradient equality, JML hard/soft/empty/ignore/detach and gradcheck away from L1 kinks; locked author JML/CWMI/DConv formula gradient parity; real CWMI pyramid constant/crop/ignore tests |
| Q / parameters | C F_prev order, scale-invariant subspace, zero spectrum deterministic basis, rank clamps, full-row-space identity, full-model native off, 3x3 padding0/bias/resize and independent-process merge parity |
| Gradients | All five nonzero U routes, A/B versus R white lists after nonzero updates, nonzero L input-factor gradients, frozen/EMA gradients absent, Adam total-update distinction |
| F2 | 16 probe full forwards + 24 VJPs per stage, no optimizer state during probes, entry function and A norm preserved, thin SVD/right projector, explicit zero-rank failure kernel, old-adapter rejection |
| F3 | Exact identity forward/custom D*g backward, no ordinary gradcheck, source-direction mapping and kappa-zero limit, observed different complementary-view scales |
| F4 | True second-order shape loss, disc union/cup mapping and stencil erosion, 64x64 d, exact zero-shape/zero-step path, deterministic finite detached target, trust bound and no teacher grads |
| F5 | Empirical set permutation and gradient checks, class support exclusion, independent sampling, equal clean-U exposure with SWD off and changed U gradients when on |
| State | Exact checkpoint replay for F1–F5; before/after optimizer/EMA and during-save injection with retained physical costs; skip transaction; shared F_prev and effective-G EMA |
| Plan / gate | Full 424-stage DAG, source/selection dependencies, cross-trajectory rejection, rank alias reporting, independent approval field/cap tests and CLI refusal before opening an approval path |
| Analysis | Three-domain formulas, unclipped negative forgetting, empty/one-empty distance policy, negative-result selection, paired patient resampling shared across seed/order/stage |

The synthetic parent is deterministic with no BN or stochastic classifier. These tests do not verify unknown real stochastic-layer behavior, hard/soft KI rules, geometry, native Dice, source identity, loader permissions or CUDA/real AMP. These remain binding/qualification items rather than being mislabeled skips or passes. Mathematical rank-insufficient fallback is tested at the kernel boundary; no performance claim is made when a future real probe falls back.
