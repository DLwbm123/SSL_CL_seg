# BPRC-X1 exploratory diagnostic

**Status:** `BPRC_X1_PREVIOUS_UTILITY_NOT_SUPPORTED`  
**Seed-0 pilot authorized:** `false`  
**Candidate:** `X1 = B2 top-2 class-balanced / 3`  
**Optimizer steps:** `0`

This is a user-authorized exploratory experiment outside the frozen BPRC V0.1 protocol. BPRC V0.1 artifacts were not changed.

## Gates

| Gate | Result | Metrics |
|---|---|---|
| Gradient scale | PASS | `{"comparisons": 192, "median_x1_to_x0": 1.9825342054338013, "nonfinite_count": 0, "p10_x1_to_x0": 1.042117820131082, "p90_x1_to_x0": 3.9574352131577566}` |
| Previous-site utility | FAIL | `{"comparisons": 192, "fraction_x1_lower_than_x0": 0.5052083333333334, "median_paired_x1_minus_x0": -2.5423243641853333e-05, "median_previous_delta": {"X0": -0.0004644850268959999, "X1": -0.000531722791492939}}` |
| Current-site safety | PASS | `{"median_current_dice_delta": {"X0": 8.690357208251953e-05, "X1": 0.00025519728660583496}, "median_current_loss_delta": {"X0": -0.00014779716730117798, "X1": -0.000570172443985939}, "x0_dice_bound": -0.0019130964279174805, "x0_loss_bound": -0.0001448412239551544}` |
| Disc-rim margin | FAIL | `{"class_median_x1_minus_x0": {"0": -1.6461910954446157e-05, "1": -7.42509572738137e-05, "2": 0.00018786472938381316}, "comparisons": 192, "fraction_x1_disc_rim_gt_x0": 0.328125, "median_disc_rim_x1_minus_x0": -7.42509572738137e-05}` |

## Decision

At least one frozen diagnostic gate failed; stop without method registration or training.
