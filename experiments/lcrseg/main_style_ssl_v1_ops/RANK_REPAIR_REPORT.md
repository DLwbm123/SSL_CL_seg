# Prostate seed164: engineering rank admission recovery

The completed RUNMC source is reused at13600 updates. Its original BMC attempt failed during stage initialization, with0 physical optimizer calls; I2CVB/HK/UCL/BIDMC were blocked. This is an engineering recovery authorized by the user on2026-10-14, not a method-selection experiment or evidence of higher segmentation accuracy.

## Cause and bounded fix

`decoder.dec3.merge.block.3.weight` is64x576. In seed164 its numerical rank is61 under the existing float64 SVD tolerance3.729000154181988e-13. The final three singular values are1.650864274578125e-13,1.2528500338216513e-13,4.1151605219150894e-14. All native tensors are finite, no row is zero, and every sealed native tensor equals its corresponding final student tensor. This supports a valid rank-deficient learned kernel, rather than a broken checkpoint or failed SVD computation.

The actual protected dimension is32. The32nd singular value is0.024327530386353706, the33rd0.02169617655058292, and the retained energy0.9998434959230295. The protection basis is therefore supported. The old full-rank requirement64 was unnecessarily restrictive for a top32 projector. Seed163 and the other13 seed164 layers passed the old full-rank check.

The main-method Parent now opts into numerical_rank>=k. Legacy callers keep require_full_rank=True. Nonfinite, zero, invalid-geometry and insufficient-protected-rank kernels still fail. No singular-value tolerance, weights, top-k computation, random initialization, projection dimension, adapter rank or optimizer/loss option changes. All healthy full-rank basis tensors compared by the regression/diagnostic are bitwise unchanged.

## Engineering evidence and negative records

- Unit admission tests passed: rank61/protected32 accepted; rank31/protected32 rejected; zero/NaN/Inf rejected; healthy RGB/mono/deeper-layer bases unchanged; legacy rank61 guard still rejects.
-67 scoped shared-engine CPU tests passed. The initial broader invocation had66 passed,1 failed because the harness ran outside the source directory, and16 fixture setup errors because legacy CWMI author-source configuration was absent. The corrected source-directory invocation explicitly deselected those16 legacy backend tests, which this main method does not use. Both logs are retained; no claim that the entire historical suite passed.
- From the existing seed164 source, target native CUDA backward, active U, nonzero A/B/R gradients, EMA update, immutable entry, merge equivalence, exact student/EMA/optimizer checkpoint replay and golden loss replay passed.
- A fixed96-update two-labeled-case BMC engineering fixture reduced mean first/last4 losses1.1288258363750934→0.09605898063139534. This is an engineering fitting check, not held-out performance. Training L labels only; no source optimizer updates and no independent test access.
-100 target engineering optimizer calls total:4 backward/resume calls plus96 fitting calls, outside formal ledgers. No engineering-trained model feeds the formal run. The first GPU reservation precheck declined to launch onGPU5 when its second formal slot became occupied;0 updates occurred. A bounded CPU coordinator pause reserved a free GPU slot for qualification; existing training workers were not paused or restarted.

## Recovery contract

Keep the frozen30-node511600-update PLAN and completed13600-update source. The five seed164 target caps remain22000/28800/6000/7200/11100, sum75100. Original BMC FAILED/EXIT1/COSTS and all four BLOCKED directories, worker config and failure log are archived under `rank_repair_20261014T024957Z/prior_attempt/` before rearming. Root attempt history and final summary reference this prior attempt even if recovery succeeds.

The replacement scheduler adopts every current worker by PID and start ticks, retains GPUs4–7 with2 slots/card, and selects the repaired code snapshot only for those five nodes. All other workers/sources keep their code and training state. The recovered BMC entry loads the original sealed RUNMC source, starts target cursor0, and executes its original22000-step cap. Source rerun0; repeated formal optimizer calls0 because the failed attempt executed none. Private checkpoints and patient artifacts remain on NAS.

At2026-10-14 03:00:02UTC, recovered BMC was RUNNING onGPU7, workerPID2054836, with100 logged committed updates and126 physical calls. Two8-second-separated samples increased its ledger97→126; the source remained13600. The recovered target ledger is contiguous1..N. The archived original worker configuration and replacement formal configuration are identical. All seven adopted worker PIDs were retained; current formal occupancy is2 workers/card onGPU4–7. CoordinatorPID2054688 supersedes2008159. Current BMC remains in its unchanged20% L-only warmup, not a disabled U path.

Algorithm fix commit`01cec42e8b7a0b96078f0cdf4c183b75e4f1d174`; operational/recovered execution commit`c234bcd23e0fc26737eb7833624434686c0846e6`. The repaired snapshot was qualified before dispatch; the later operational commit adds only the recovery operator. Exact startup evidence is in `RANK_REPAIR_STARTUP.json`; private qualification/diagnostic, archives, provenance, handoff, history and health receipts are retained under the study root's repair directory. Before dispatch, the operator's provenance-recording template incorrectly substituted a filename token and failed with FileNotFoundError. This was corrected before coordinator replacement, with0 worker restarts or formal optimizer calls; the error is preserved in `PRE_DISPATCH_ERROR.json`.

Formal training is ongoing; no scientific gain or completed experiment delivery is claimed. Source/report changes are committed locally; public result delivery remains due when the experiment actually completes.
