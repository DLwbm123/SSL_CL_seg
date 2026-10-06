# Stage A — completed; action signal passes

All 144 branches completed: 8 contexts × 9 actions × 2 paired student streams × 100 updates. Exactly 4/8 contexts passed the frozen engineering gate. All passing contexts use 2 support images; none of the 8-image contexts passed. These are image-level exploratory results, not patient-independent or statistical confirmation.

| Context | Passing non-native actions |
|---|---|
| m2000_n2_brightness | 3, 6 |
| m2000_n2_contrast | 2, 3, 4, 5, 6, 7, 8 |
| m2000_n8_brightness | none |
| m2000_n8_contrast | none |
| m8000_n2_brightness | 1, 2, 4, 5, 7, 8 |
| m8000_n2_contrast | 5, 8 |
| m8000_n8_brightness | none |
| m8000_n8_contrast | none |

The global FIXED_BEST action is 8: alpha=0.5; background/rim/cup weights=(0.5,1.0,1.5). It was selected from all contexts and both Q_train streams using the frozen gain-minus-retention-penalty utility. No target-domain outcomes were read.

The paired-repeat reward noise floor is 0.002027520. This is a repeat fluctuation scale, not a standard error.

| Action | Mean utility (×100) |
|---|---|
| 0 | 0.000428 |
| 1 | -0.108863 |
| 2 | 0.056990 |
| 3 | 0.112214 |
| 4 | 0.021495 |
| 5 | 0.143039 |
| 6 | 0.197870 |
| 7 | 0.102268 |
| 8 | 0.208791 |

Utility is not a Dice difference. Complete image-macro rim/cup results, short/final readouts, losses, gate coverage and parameter differences are in ACTION_ROWS.jsonl; all paired differences, including adverse effects, are in ACTION_PAIRED_RESULTS.csv.

New physical cost through A: 23,221 student updates (21 qualification including repairs, 8,000 auxiliary, 800 entries, 14,400 audit) and 4 actor qualification updates. The designated common REFUGE checkpoint has 8,000 additional historical updates. No prior training or target performance endpoint has run yet.

Decision: proceed to implementation and qualification of stage B; this pass does not establish GRPO value.
