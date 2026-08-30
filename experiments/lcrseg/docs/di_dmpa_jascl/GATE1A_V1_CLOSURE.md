# Gate 1A v1 closure

Date: 2026-08-30, Asia/Shanghai. Branch `codex/gate1a-v2-null-aware-sphere` starts from `606a5c53a37d0e4c9605415e8b38a1f177d1604f`, not main.

## Closure decision

**CLOSED_V1_FEATURE_SUPPORT_ASSUMPTION_FALSIFIED**.

The v1 assumption `all registered decoder.dec1 post-ReLU 16-D features have norm > 1e-12` is **FALSIFIED**. Root cause: **POST_RELU_FEATURE_SUPPORT_INCLUDES_EXACT_ZERO_ATOM**. This is not another guard implementation defect. No third v1 attempt is authorized.

The exact observed registered zero is B0-EMA / seed2 / stage0 / REFUGE / train_labeled / REFUGE_test_n0128 / class1 / (y=125,x=212), norm=0. Checkpoint B0/seed2/stage0 SHA256: `244c87368f252a660bf0d1934bf0ccf512790dc698d04ede01196d14c34064ac`; sampling unit SHA256: `5574b03230f7e724747174387edd1e96c70f565d52cfb61f9a53414ceb57a8d9`.

## Immutable history

| Identity | Commit |
| --- | --- |
| v1 preregistration | `cfb62554f1e6a2a36850547485b1857dc9a28a20` |
| attempt1 code | `8f4a71a5ea8d145183a3007ccd398ab79387478e` |
| attempt1 report | `945b484072cb9f2757be98df34e5d72844596e84` |
| scope clarification | `e8336da9d7364f4b67912d03791195445318afc3` |
| attempt2 code | `a89716ddbd2eccbe76c574e97e520d424aa923ab` |
| attempt2 report | `606a5c53a37d0e4c9605415e8b38a1f177d1604f` |

Both attempts permanently retain BLOCKED_NUMERICAL_FAILURE, geometry_jobs=0, A1–A6 uncomputed and selected_K=null. No mechanism geometry results have been observed. All preregistration, attempt and report bytes remain untouched; this closure is a separate additive record, not a rewrite of either machine state.

Frozen sampling-plan raw SHA256: `96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`.

Next protocol: **GATE1A_V2_NULL_AWARE_SPHERE**. Its preregistration and separate execution authorization must both be published and remotely verified before new checkpoint-tensor reads or model forwards. Model/transport optimizer steps remain0; no method registration, training, downstream Gate1B/C or main merge.
