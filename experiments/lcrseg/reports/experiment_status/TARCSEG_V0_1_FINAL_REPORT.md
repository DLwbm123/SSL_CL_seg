# TARC-Seg V0.1 final report

**Final status:** `TARC_RELATION_FIDELITY_NOT_SUPPORTED`  
**Protocol consequence:** hard stop before Part B method implementation or training  
**Optimizer steps executed by TARC:** `0`

## 1. ASPR freeze and relation-space audit

`ASPR_V0_1_FROZEN_FOR_TARC` was recorded before TARC work. The freeze preserves the ASPR result `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`: Gates A/D failed, Gates B/C passed, H3 was not promoted, and neither unlabeled memory nor site modes were enabled.

`TARC_RELATION_SPACE_AUDIT_PASSED` then confirmed:

- relation source `UNet2D dec3 -> existing ProjectionHead`;
- relation dimension 128 and existing quarter-resolution grid;
- class order `0 background`, `1 optic_disc_rim`, `2 optic_cup`;
- relation temperature 0.1 and existing strict relation valid mask;
- all nine R0 site checkpoints and all six old/current pairs;
- all-class transport, including background, with no foreground-only fallback;
- current-site `train_labeled` as the sole transport-fitting view.

## 2. Part-A implementation

Added only feasibility infrastructure:

- generic all-class case-prototype and shrinkage transport utilities;
- immutable serializable epoch transport state;
- frozen R0 feature/anchor audit helpers;
- anchor transport, relation fidelity/current safety, virtual-step, and gate compiler scripts;
- seven transport primitive tests.

The existing ASPR shrinkage formula was retained. Each eligible case contributes one normalized class prototype; every class requires at least 32 relation pixels; `n < 2`, non-finite, or zero-signal estimates produce zero transport. Background is treated identically as a required class. Historical anchors are never mutated.

## 3. Gate A — all-class anchor transport

**Result: PASS.**

- median class-transport minus static oracle cosine: `+0.824633` (required `>= +0.020`);
- class transport better than static: `100%` of 18 seed/transition/class pairs (required `>=70%`);
- 10th-percentile improvement: `+0.251887` (required `>=-0.020`);
- class-positive seed counts: background `3/3`, disc-rim `3/3`, cup `3/3`;
- background median improvement: `+0.943351` (required `>=+0.010`);
- every transport had at least two paired cases; historical anchors were byte-identical to the frozen prior-site anchors.

## 4. Gate B — historical relation fidelity

**Result: FAIL.**

Most aggregate criteria passed:

- median KL reduction versus static: `96.60%` (required `>=10%`);
- class KL lower than static: `100%` of six seed/transition pairs;
- median top-1 agreement improvement: `+0.899517`;
- median class KL `0.157217` versus global KL `4.001424`;
- median class top-1 agreement `0.954127` versus global `0.057243`;
- non-finite count `0`.

The classwise margin-agreement safety condition failed. Median deltas versus static were:

| Class | Median margin-agreement delta |
|---|---:|
| background | `+0.006729` |
| optic-disc-rim | `-0.030954` |
| optic-cup | `+0.025737` |

The disc-rim drop is below the allowed `-0.010`, so the exact first failed research state is `TARC_RELATION_FIDELITY_NOT_SUPPORTED`.

## 5. Gate C — current-site safety

**Result: FAIL independently.**

- minimum class-transport accuracy delta versus static: `+0.023766` (passes `>=-0.005`);
- minimum margin delta: `-0.011178` (fails `>=-0.010`);
- median accuracy delta: `+0.906425`;
- median margin delta: `+0.008812`;
- non-finite count: `0`.

## 6. Gate D — functional virtual step

**Result: FAIL independently.**

The audit used exactly 32 current-site update batches, 16 previous-site validation batches, and 16 current-site validation batches per transition with `virtual_step_norm=1e-3`.

- T3 previous-val delta better than T0: `39.58%` of 192 comparisons (required `>=60%`);
- median previous-val delta T0: `-0.00167272`;
- median previous-val delta T2: `-0.00106583`;
- median previous-val delta T3: `-0.00165457`;
- T3 did not beat T0 by the required `1e-4` median margin;
- the current-site and T3-vs-T2 conditions passed;
- old-model gradient count was zero;
- all current-model SHA-256 values were identical before and after every audit transition.

## 7. Engineering verification

- relation-space audit: passed;
- 18 anchor rows, 48 fidelity/safety rows, and 768 virtual-step rows: exact;
- T0/T1/T2/T3 virtual rows: 192 each;
- transport classes: exactly `{0,1,2}`;
- optimizer steps: zero;
- hidden-GT training/transport use: none;
- complete project regression: `183/183` passed;
- TARC method/config/run targets: absent.

One seed-0 CSV schema defect was caught before gate compilation: the first writer schema omitted current-safety-only columns. The malformed CSV and premature summary were preserved under `reports/failures/tarc_relation_fidelity_seed0_schema_bug_20260828T132900Z.*`; the corrected run changed only the explicit CSV column union, not data, formulas, thresholds, or checkpoints.

## 8. Canonical evidence

- freeze: `reports/experiment_status/ASPR_V0_1_FREEZE_FOR_TARC.{md,json}`
- relation audit: `reports/experiment_status/TARC_RELATION_SPACE_AUDIT.{md,json}`
- feasibility decision: `reports/experiment_status/TARC_FEASIBILITY_AUDIT.{md,json}`
- anchor audit: `reports/analysis/tarcseg_v0_1/anchor_transport_audit.csv`
- relation/safety audit: `reports/analysis/tarcseg_v0_1/relation_fidelity_audit.csv`
- virtual-step audit: `reports/analysis/tarcseg_v0_1/virtual_step_audit.csv`

Canonical SHA-256 values:

- feasibility JSON: `af6c12dd706299c7b51c721487a9ad6ebba0a93903cc243d82ea4aec4d4141c1`
- feasibility Markdown: `9b4ebabe6dc610d5374f2d2f06767e193d4a552c9a4bd34396b558a6ad055a63`
- anchor CSV: `0dc032c7c7792d7dc8510063cbdd983330b8301081fc2eadca45e8861a1e2892`
- relation CSV: `4c870c22bd4658a89ea71f81fdaae7bb613e3755e4c678e8399b6bb4c32c00ca`
- virtual-step CSV: `7a5f2c7c0882d761be80e37da95850308e06af4da04ae98b0b56a79d0b16eae9`

## 9. Explicitly unexecuted

Because the Part-A status is not `TARC_FEASIBILITY_SUPPORTED`, the following were prohibited and not executed:

- TARC loss and `tarcseg_v0_1` training method;
- T0/T1/T2/T3/shift-swap training configs;
- T0 one-batch or 500-step equivalence bridge;
- seed-0/seed-1 pilots;
- seed-0 or multi-seed full runs;
- shift-swap, external baselines, Prostate, or M&Ms experiments.

No completed run was rerun or overwritten, and no hyperparameter was changed.
