# BPRC-Seg V0.1 feasibility audit

**Final status:** `BPRC_GRADIENT_SCALE_NOT_SUPPORTED`  
**Part B authorized:** `false`  
**Optimizer steps:** `0`  
**Hidden GT training usage:** `none`

## Engineering boundary

- TARC remained frozen at `TARC_RELATION_FIDELITY_NOT_SUPPORTED`.
- The audit reused the exact frozen TARC metric functions and R0 objective path.
- B0/B1/B2/B3 were evaluated on fixed 32-update/16-previous-val/16-current-val batches for each of two transitions and three seeds.
- Functional updates were stateless with norm `1e-3`; optimizer steps, checkpoint mutation, old-model gradients, and hidden-GT training usage were zero.
- Boundary/interior and class diagnostic labels were used post-hoc only.

## Gate results

| Gate | Result | Metrics |
|---|---|---|
| Gradient scale | FAIL | `{"comparisons": 192, "median_b3_to_b0_gradient_norm_ratio": 3.5308377735091927, "nonfinite_count": 0, "p10_b3_to_b0_gradient_norm_ratio": 1.956722571345011, "p90_b3_to_b0_gradient_norm_ratio": 6.808759859258404}` |
| B3 vs B0 previous-site utility | FAIL | `{"comparisons": 192, "fraction_b3_delta_lower_than_b0": 0.515625, "median_previous_loss_delta": {"B0": -0.0004639141261577606, "B1": -0.0005124900490045547, "B2": -0.0006722845137119293, "B3": -0.00036555808037519455}}` |
| Current-site safety | PASS | `{"b0_dice_minus_0_002_bound": -0.001913051724433899, "b0_loss_2pct_safety_bound": -0.00014498360455036163, "median_current_dice_delta": {"B0": 8.694827556610107e-05, "B1": 0.0002874806523323059, "B2": 0.00028942152857780457, "B3": 0.0002924315631389618}, "median_current_loss_delta": {"B0": -0.000147942453622818, "B1": -0.0006854701787233353, "B2": -0.0006698807701468468, "B3": -0.0006766896694898605}}` |
| Disc-rim margin | FAIL | `{"class_median_b3_minus_b0": {"0": -2.200681587438691e-05, "1": -5.80597434472474e-05, "2": 0.00019748651775486348}, "comparisons": 192, "fraction_b3_disc_rim_margin_gt_b0": 0.4270833333333333, "median_disc_rim_b3_minus_b0": -5.80597434472474e-05}` |
| B3 beyond B1 class balance | FAIL | `{"disc_rim_b3_minus_b1": 0.00010106711859769746, "median_disc_rim_margin_agreement_after": {"B0": 0.7797529219077834, "B1": 0.7793142041815522, "B2": 0.7792655504959956, "B3": 0.7794152713001499}, "median_previous_loss_delta": {"B0": -0.0004639141261577606, "B1": -0.0005124900490045547, "B2": -0.0006722845137119293, "B3": -0.00036555808037519455}}` |
| B3 all competitors beyond B2 top-2 | FAIL | `{"class_median_margin_b3_minus_b2": {"0": 5.629075462032507e-06, "1": 8.400508509248672e-05, "2": -4.630271646227646e-05}, "median_previous_loss_delta": {"B0": -0.0004639141261577606, "B1": -0.0005124900490045547, "B2": -0.0006722845137119293, "B3": -0.00036555808037519455}}` |

## Protocol decision

Protocol hard stop before BPRC method/config implementation or training. STOP_NEW_RELATION_METHODS is binding.

## Canonical artifacts

- `reports/analysis/bprcseg_v0_1/feasibility_gradient_scale.csv`
- `reports/analysis/bprcseg_v0_1/feasibility_virtual_steps.csv`
- `reports/analysis/bprcseg_v0_1/feasibility_margin_analysis.csv`
- `reports/experiment_status/BPRC_FEASIBILITY_AUDIT.json`
