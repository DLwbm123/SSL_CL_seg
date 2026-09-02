# RC-SHOR V0.5 public erratum

RC-SHOR V0.5's scientific status remains **FAIL_RC_SHOR_VALUE**. C6 remains exactly C0;
all previously published C0-C8 Dice values and the C3-C6 direct comparison remain unchanged.

The C3/C4/C5 routing summaries in the original `RC_SHOR_V0_5_ROUTING.csv` are incorrect
because fold-order route arrays were paired with global-row-order utilities. The corrected
three-domain gains are C3 0.21708218085929243, C4 0.11330836336206514, and C5 0.020747627739757475; C6 remains 0.

The original stability p10/p90 values were computed from auxiliary single-bootstrap routes,
not the full final-C6 consensus procedure: those routes ignored `rho`, the feasible >= 90 gate,
and ensemble consensus. The reported 5/100 is the intersection of feasible replicate indices
across five outer folds, not the preregistered per-unit feasibility measure.

This erratum does not rerun or re-adjudicate V0.5 and does not alter any original V0.5 byte.
