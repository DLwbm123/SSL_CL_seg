# PPC-SHOR V0.6A final report

## Outcome

The sole authorized attempt stopped before segmentation evaluation. The final engineering
status is **`BLOCKED_PROTOCOL_OR_LEAKAGE`**, not a scientific failure and not a valid
design-degeneracy result. No retry is authorized by this protocol.

The raw executor wrote `BLOCKED_DESIGN_DEGENERATE_BEFORE_GT` after two gate failures. A
derived-artifact-only postflight audit found that both were false negatives:

- Every outer case had 200/200 finite predictions from each historical calibrator. The gate
  read `route_policy`'s route-eligibility denominator, which deliberately remains zero when
  ridge top1 is the current expert, instead of the calibrator prediction-validity denominator.
- The reported 0.929–0.989 parameter ratios counted adjacent PAV blocks with identical fitted
  probabilities as separate free parameters. Coalescing equal fitted levels gives fold maxima
  0.07792, 0.08108, 0.08642, 0.07692, and 0.08108; the overall maximum is 0.08642 <= 0.10.

Thus all 11 design conditions are supported by the sealed GT-free artifacts, but the frozen
executor had already consumed the one attempt and stopped. The raw preflight JSON is preserved
unchanged for auditability; this report supplies the higher-priority protocol-blocked status.

## Population and calibration

The frozen population contains 990 calibration seed-rows from 575 patients and 198 own-seed
`train_labeled` segmentation seed-rows from 177 patients. Inner folds contained 919–926 rows
and 539–540 patients; outer folds contained 37–42 rows and 35–36 patients. Pooled active
support was 326–334 patients for expert 0 and 134–141 for expert 1. Every fold/expert completed
200/200 feasible Bayesian fits. Respectively 8, 8, 8, 11, and 11 of 12 candidates passed the
calibration gates before the stop, but no candidate was formally selected.

The 12 candidate route frequencies ranged from 0.67677 to 0.80303 and formed 10 distinct route
arrays. Exact candidate calibration rows, frequencies, route hashes, duplicates, fold counts,
and all five pre-seals are public in the CSV/JSON artifacts.

## What was not evaluated

C0–C8 Dice, value, safety, full-policy stability, jackknife, and ordinary-bootstrap sensitivity
are all unevaluated (`null`), not zero. No expert prediction cache was materialized, so model
forwards were also zero. Segmentation and router optimizer/update counts, outer GT/domain reads,
V0.4 `formal_03` reads, main merges, and external tests were all zero.

The server freeze suite passed 10/10 tests; 16/16 predecessor compatibility tests also passed
locally. The private GT-free attempt inventory is 21 files / 44,556,824 bytes; no private raw
artifact is published.
