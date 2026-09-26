# DEVELOPMENT completion

Status: **R1_5_DEVELOPMENT_NOT_MET**. Endpoints: 20/20. Commits: 20000/20000. Attempts: 21564; failed: 0; replay: 1564.

## Primary A4−A0

| Seed | Backbone | Domain | Macro | Rim | Cup |
|---|---|---|---:|---:|---:|
| 261 | DINOV2_VITS14_QUERY | Drishti_GS | +0.000546 | +0.000311 | +0.000781 |
| 261 | DINOV2_VITS14_QUERY | RIM_ONE_r3 | -0.000780 | +0.000572 | -0.002131 |
| 261 | UNET_QUERY_128 | Drishti_GS | -0.002535 | -0.004860 | -0.000210 |
| 261 | UNET_QUERY_128 | RIM_ONE_r3 | -0.000938 | -0.001148 | -0.000728 |

## Frozen gate

{
  "backbone_means": {
    "DINOV2_VITS14_QUERY": {
      "macro": -0.00011678446084262273,
      "rim": 0.0004414412379265542,
      "cup": -0.0006750101596116886
    },
    "UNET_QUERY_128": {
      "macro": -0.0017365627642720871,
      "rim": -0.0030042079091071594,
      "cup": -0.0004689176194369038
    }
  },
  "primary_mean": -0.0009266736125573549,
  "positive_cells": 1,
  "worst_macro_cell": -0.002535221576690727,
  "class_violations": [],
  "passes": false
}

**Class loss alerts:** NONE

The full matrix completed before the progression decision. The conservative preregistered class-mean floor is zero. No quotas, weights, ramps, seeds or arms were selected after viewing results. R1 results remain unchanged.

## Interpretation

Factor effects for macro, rim and cup are in FACTOR_EFFECTS.csv (development). Per-cell contrasts include A1/A2/A3/A4 versus A0 and A2/A3/A4 versus A1. Mechanism diagnostics retain every 100-update record; compare early clipping and gradient cosines without inferring causal evidence beyond the registered contrasts.

Single development seed and exposed validation are not independent patient confirmation or statistical significance. Confirmation uses new training seeds on the same exposed validation; it is not an independent dataset. No R2/R3 starts automatically.

See [FINAL_INTERPRETATION.md](FINAL_INTERPRETATION.md) for all 15 registered questions, reproduction checks, repair accounting and final audited state.
